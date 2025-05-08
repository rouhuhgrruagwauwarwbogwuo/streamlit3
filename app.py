# ✅ 修改說明：將 Logistic Regression 直接改為載入預先訓練好的模型，避免每次都重新訓練錯誤。
# ✅ 使用 joblib 載入訓練好的 stacking 邏輯回歸模型
# ✅ 修復 Xception 模型輸入尺寸不同可能導致的錯誤
# ✅ 將模型與邏輯回歸只載入一次，提升效能

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
import joblib

@st.cache_resource
def load_models():
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    xception_model = Xception(weights='imagenet', include_top=False, pooling='avg', input_shape=(299, 299, 3))

    resnet_classifier = Sequential([resnet_model, Dense(1, activation='sigmoid')])
    efficientnet_classifier = Sequential([efficientnet_model, Dense(1, activation='sigmoid')])
    xception_classifier = Sequential([xception_model, Dense(1, activation='sigmoid')])

    return {
        'ResNet50': resnet_classifier,
        'EfficientNet': efficientnet_classifier,
        'Xception': xception_classifier
    }

@st.cache_resource
def load_stacking_model():
    url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/stacking_model.pkl"
    local_filename = "stacking_model.pkl"
    if not os.path.exists(local_filename):
        r = requests.get(url)
        with open(local_filename, 'wb') as f:
            f.write(r.content)
    return joblib.load(local_filename)

def extract_face(pil_img):
    img_array = np.array(pil_img)
    faces = detector.detect_faces(img_array)
    if len(faces) > 0:
        x, y, w, h = faces[0]['box']
        face = img_array[y:y+h, x:x+w]
        return Image.fromarray(face)
    return None

def preprocess_image(img, model_name):
    if model_name == 'Xception':
        img_resized = img.resize((299, 299))
    else:
        img_resized = img.resize((224, 224))

    img_array = np.array(img_resized).astype(np.float32)

    if model_name == 'ResNet50':
        return preprocess_resnet(img_array)
    elif model_name == 'EfficientNet':
        return preprocess_efficientnet(img_array)
    elif model_name == 'Xception':
        return preprocess_xception(img_array)
    return img_array

def predict_model(models, img):
    predictions = []
    for model_name, model in models.items():
        preprocessed_img = preprocess_image(img, model_name)
        prediction = model.predict(np.expand_dims(preprocessed_img, axis=0), verbose=0)
        predictions.append(prediction[0][0])
    return predictions

def stacking_predict(models, img, stacking_model):
    if not models or not stacking_model:
        return "模型加載失敗"

    predictions = predict_model(models, img)
    stacked = np.array(predictions).reshape(1, -1)
    final_prediction = stacking_model.predict(stacked)[0]
    return "Deepfake" if final_prediction == 1 else "Real"

def show_prediction(img, models, stacking_model):
    label = stacking_predict(models, img, stacking_model)
    st.image(img, caption="輸入圖像", use_container_width=True)
    st.subheader(f"集成模型預測結果: **{label}**")

st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖像偵測器")

detector = MTCNN()
models = load_models()
stacking_model = load_stacking_model()

tab1, tab2 = st.tabs(["🖼️ 圖像偵測", "🎥 影片偵測"])

with tab1:
    st.header("上傳圖像進行 Deepfake 偵測")
    uploaded_image = st.file_uploader("選擇一張圖像", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="原始圖像", use_container_width=True)
        face_img = extract_face(pil_img)
        if face_img:
            st.image(face_img, caption="偵測到的人臉", width=300)
            show_prediction(face_img, models, stacking_model)
        else:
            st.info("⚠️ 沒有偵測到人臉，使用整張圖進行預測")
            show_prediction(pil_img, models, stacking_model)

with tab2:
    st.header("影片偵測（處理前幾幀）")
    uploaded_video = st.file_uploader("選擇一段影片", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

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
                    show_prediction(face_img, models, stacking_model)
                    shown = True
            frame_idx += 1
        cap.release()
