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

# 下載模型（如果需要的話）
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
            print(f"下載失敗，狀態碼: {response.status_code}")
            return None
    return model_filename

# 載入模型
def load_models():
    try:
        resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        xception_model = Xception(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

        # 自訂分類器
        resnet_classifier = Sequential([resnet_model, Dense(1, activation='sigmoid')])
        efficientnet_classifier = Sequential([efficientnet_model, Dense(1, activation='sigmoid')])
        xception_classifier = Sequential([xception_model, Dense(1, activation='sigmoid')])

        return {
            'ResNet50': resnet_classifier,
            'EfficientNet': efficientnet_classifier,
            'Xception': xception_classifier
        }
    except Exception as e:
        print(f"載入模型時出錯: {e}")
        return None

# MTCNN 初始化
detector = MTCNN()

# 提取人臉
def extract_face(pil_img):
    img_array = np.array(pil_img)
    faces = detector.detect_faces(img_array)
    if len(faces) > 0:
        x, y, w, h = faces[0]['box']
        face = img_array[y:y+h, x:x+w]
        return Image.fromarray(face)
    return None

# 圖像預處理
def preprocess_image(img, model_name):
    img_resized = img.resize((224, 224))  # 對所有模型進行 224x224 重設
    img_array = np.array(img_resized).astype(np.float32) / 255.0  # 標準化 RGB 圖像

    if model_name == 'ResNet50':
        return preprocess_resnet(img_array)
    elif model_name == 'EfficientNet':
        return preprocess_efficientnet(img_array)
    elif model_name == 'Xception':
        img_resized = img.resize((299, 299))  # Xception 要求 299x299 大小
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        return preprocess_xception(img_array)
    return img_array

# 模型預測
def predict_model(models, img):
    predictions = []
    for model_name, model in models.items():
        preprocessed_img = preprocess_image(img, model_name)
        prediction = model.predict(np.expand_dims(preprocessed_img, axis=0))  # 增加批次維度
        predictions.append(prediction[0][0])  # 扁平化預測結果
    return predictions

# 堆疊預測（集成學習）
def stacking_predict(models, img):
    if not models:  # 如果模型字典為空，返回錯誤訊息
        return "模型加載失敗"
    
    predictions = predict_model(models, img)
    
    # 確保預測結果是 2D 數組，並根據需要重塑
    stacked_predictions = np.array(predictions).reshape(1, -1)  # (1, n_models)
    print("堆疊預測:", stacked_predictions)  # 調試輸出

    # 創建標籤（Deepfake 為 1，Real 為 0）
    labels = [1 if p > 0.5 else 0 for p in predictions]
    print("標籤:", labels)  # 調試輸出
    
    labels = np.array(labels).reshape(-1, 1)  # 確保標籤是 2D 數組 (n_samples, 1)
    print("重塑後標籤:", labels)  # 調試輸出

    try:
        # 分割資料為訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(stacked_predictions, labels, test_size=0.2, random_state=42)
        
        logistic_model = LogisticRegression()
        logistic_model.fit(X_train, y_train.ravel())  # 把 y_train 壓平成一維

        # 在測試集上進行預測
        final_prediction = logistic_model.predict(X_test)[0]
        print("最終預測:", final_prediction)  # 調試輸出
        return "Deepfake" if final_prediction == 1 else "Real"
    except ValueError as e:
        print(f"訓練邏輯回歸模型時出錯: {e}")
        return "模型訓練失敗"

# 顯示預測結果
def show_prediction(img, models):
    label = stacking_predict(models, img)
    st.image(img, caption="輸入圖像", use_container_width=True)
    st.subheader(f"集成模型預測結果: **{label}**")

# Streamlit UI
st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖像偵測器")

tab1, tab2 = st.tabs(["🖼️ 圖像偵測", "🎥 影片偵測"])

# 圖像偵測
with tab1:
    st.header("上傳圖像進行 Deepfake 偵測")
    uploaded_image = st.file_uploader("選擇一張圖像", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="原始圖像", use_container_width=True)

        face_img = extract_face(pil_img)
        if face_img:
            st.image(face_img, caption="偵測到的人臉", width=300)
            models = load_models()
            show_prediction(face_img, models)
        else:
            st.info("⚠️ 沒有偵測到人臉，使用整張圖進行預測")
            models = load_models()
            show_prediction(pil_img, models)

# 影片偵測（只處理前幾幀）
with tab2:
    st.header("影片偵測（處理前幾幀）")
    uploaded_video = st.file_uploader("選擇一段影片", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        import cv2
        st.info("🎬 正在提取幀並分析...")
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
                    st.image(face_img, caption=f"第 {frame_idx} 幀 偵測到的人臉", width=300)
                    models = load_models()
                    show_prediction(face_img, models)
                    shown = True
            frame_idx += 1

        cap.release()
