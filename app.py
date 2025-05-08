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
from keras.preprocessing.image import img_to_array
from mtcnn import MTCNN
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

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

# ✅ 載入模型
def load_models():
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    xception_model = Xception(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

    base_models = {
        "ResNet50": resnet_model,
        "EfficientNet": efficientnet_model,
        "Xception": xception_model
    }
    return base_models

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
def preprocess_image(pil_img, model_name):
    img = pil_img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if model_name == "ResNet50":
        return resnet_model.preprocess_input(img_array)
    elif model_name == "EfficientNet":
        return efficientnet_model.preprocess_input(img_array)
    elif model_name == "Xception":
        return xception_model.preprocess_input(img_array)
    return img_array

# ✅ 預測
def predict_model(models, img):
    predictions = []
    for model_name, model in models.items():
        processed_img = preprocess_image(img, model_name)
        pred = model.predict(processed_img)
        predictions.append(pred)
    return predictions

# ✅ Stacking: 輸出模型預測作為特徵，並使用 Logistic Regression 作為最終分類器
def stacking_predict(models, img):
    predictions = predict_model(models, img)
    stacked_features = np.hstack(predictions)  # 合併各模型的預測結果
    stacked_features = stacked_features.reshape(1, -1)
    classifier = LogisticRegression()
    classifier.fit(stacked_features, [0])  # 用於擬合訓練數據，這裡僅為示範
    return classifier.predict(stacked_features)

# ✅ Boosting: 使用 XGBoost 作為提升模型
def boosting_predict(models, img):
    predictions = predict_model(models, img)
    stacked_features = np.hstack(predictions)
    stacked_features = stacked_features.reshape(1, -1)
    xg_model = xgb.XGBClassifier()
    xg_model.fit(stacked_features, [0])  # 擬合模型
    return xg_model.predict(stacked_features)

# ✅ Bagging: 使用 RandomForest 來進行 Bagging
def bagging_predict(models, img):
    predictions = predict_model(models, img)
    stacked_features = np.hstack(predictions)
    stacked_features = stacked_features.reshape(1, -1)
    rf_model = RandomForestClassifier()
    rf_model.fit(stacked_features, [0])  # 擬合模型
    return rf_model.predict(stacked_features)

# ✅ 顯示預測
def show_prediction(img, models):
    label = stacking_predict(models, img)
    st.image(img, caption="輸入圖片", use_container_width=True)
    st.subheader(f"預測結果：**{label}**")

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
            show_prediction(face_img, load_models())
        else:
            st.info("⚠️ 未偵測到人臉，將使用整張圖片進行預測")
            show_prediction(pil_img, load_models())

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
                    show_prediction(face_img, load_models())
                    shown = True
            frame_idx += 1

        cap.release()
