import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
import cv2
import tempfile
from keras.applications import ResNet50, EfficientNetB0, Xception
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.resnet50 import preprocess_input as preprocess_resnet
from keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from keras.applications.xception import preprocess_input as preprocess_xception
from mtcnn import MTCNN
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ⬇️ 下載模型（如果還沒下載）
def download_model():
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    model_filename = "deepfake_cnn_model.h5"
    if not os.path.exists(model_filename):
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_filename, "wb") as f:
                f.write(response.content)
            print("模型下載成功！")
        else:
            print(f"下載失敗，狀態碼：{response.status_code}")
            return None
    return model_filename

# ✅ 載入 ResNet50、EfficientNet 和 Xception 模型
def load_models():
    try:
        resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        xception_model = Xception(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

        # 自定義分類器
        resnet_classifier = Sequential([resnet_model, Dense(1, activation='sigmoid')])
        efficientnet_classifier = Sequential([efficientnet_model, Dense(1, activation='sigmoid')])
        xception_classifier = Sequential([xception_model, Dense(1, activation='sigmoid')])

        return {
            'ResNet50': resnet_classifier,
            'EfficientNet': efficientnet_classifier,
            'Xception': xception_classifier
        }
    except Exception as e:
        print(f"載入模型錯誤：{e}")
        return None

# ✅ MTCNN 初始化
detector = MTCNN()

# ✅ 擷取人臉
def extract_face(pil_img):
    img_array = np.array(pil_img)
    faces = detector.detect_faces(img_array)
    if len(faces) > 0:
        x, y, w, h = faces[0]['box']
        face = img_array[y:y+h, x:x+w]
        return Image.fromarray(face)
    return None

# ✅ 圖片預處理
def preprocess_image(img, model_name):
    img_array = np.array(img)
    
    # 調整圖像大小
    img_resized = img.resize((224, 224))  # 所有模型都需要 224x224 大小的圖片
    img_array = np.array(img_resized)

    img_array = img_array.astype(np.float32) / 255.0  # 處理 RGB 圖像
    
    if model_name == 'ResNet50':
        return preprocess_resnet(img_array)
    elif model_name == 'EfficientNet':
        return preprocess_efficientnet(img_array)
    elif model_name == 'Xception':
        return preprocess_xception(img_array)
    return img_array

# ✅ 預測
def predict_model(models, img):
    predictions = []
    for model_name, model in models.items():
        preprocessed_img = preprocess_image(img, model_name)
        prediction = model.predict(np.expand_dims(preprocessed_img, axis=0))[0][0]
        predictions.append(prediction)
    return predictions

# ✅ Stacking 預測（集成學習）
def stacking_predict(models, img):
    predictions = predict_model(models, img)
    
    # 確保 predictions 是一維陣列，並轉換為 2D 陣列（每個模型一行）
    stacked_predictions = np.array(predictions).reshape(-1, 1)

    # 設定每個預測結果的標籤（1表示Deepfake，0表示Real）
    labels = [1 if p > 0.5 else 0 for p in predictions]
    
    # 使用邏輯回歸進行預測融合
    logistic_model = LogisticRegression()
    logistic_model.fit(stacked_predictions, labels)

    # 輸出最終預測
    final_prediction = logistic_model.predict(stacked_predictions)[0]
    return "Deepfake" if final_prediction == 1 else "Real"

# ✅ 顯示預測
def show_prediction(img, models):
    label = stacking_predict(models, img)
    st.image(img, caption="輸入圖片", use_container_width=True)
    st.subheader(f"集成模型判斷：**{label}**")

# ✅ Streamlit UI
st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖片偵測器")

tab1, tab2 = st.tabs(["🖼️ 圖片偵測", "🎥 影片偵測"])

# ✅ 圖片偵測
with tab1:
    st.header("上傳圖片進行 Deepfake 偵測")
    uploaded_image = st.file_uploader("請選擇圖片", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="原始圖片", use_container_width=True)

        face_img = extract_face(pil_img)
        if face_img:
            st.image(face_img, caption="偵測到人臉", width=300)
            models = load_models()
            show_prediction(face_img, models)
        else:
            st.info("⚠️ 未偵測到人臉，將使用整張圖片進行預測")
            models = load_models()
            show_prediction(pil_img, models)

# ✅ 影片偵測（僅擷取前幾幀）
with tab2:
    st.header("影片偵測（僅擷取前幾幀進行判斷）")
    uploaded_video = st.file_uploader("請上傳影片", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        import cv2
        st.info("🎬 擷取幀與分析中...")
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        shown = False

        while cap.isOpened() and not shown:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 10 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb)
                face_img = extract_face(pil_frame)
                if face_img:
                    st.image(face_img, caption=f"第 {frame_idx} 幀偵測到人臉", width=300)
                    models = load_models()
                    show_prediction(face_img, models)
                    shown = True
            frame_idx += 1

        cap.release()
