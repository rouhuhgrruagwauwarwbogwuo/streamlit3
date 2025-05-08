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

# 初始化 MTCNN
st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖像偵測器")
detector = MTCNN()

# 載入模型
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

# 提取人臉
@st.cache_data(show_spinner=False)
def extract_face(pil_img):
    img_array = np.array(pil_img)
    faces = detector.detect_faces(img_array)
    if faces:
        x, y, w, h = faces[0]['box']
        face = img_array[y:y+h, x:x+w]
        return Image.fromarray(face)
    return None

# 預處理圖像
def preprocess_image(img, model_name):
    if model_name == 'Xception':
        img = img.resize((299, 299))
        img_array = np.array(img).astype(np.float32)
        return preprocess_xception(img_array)
    else:
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32)
        if model_name == 'ResNet50':
            return preprocess_resnet(img_array)
        elif model_name == 'EfficientNet':
            return preprocess_efficientnet(img_array)
    return img_array

# 單模型預測
def predict_model(models, img):
    predictions = []
    for name, model in models.items():
        input_data = preprocess_image(img, name)
        prediction = model.predict(np.expand_dims(input_data, axis=0), verbose=0)
        predictions.append(prediction[0][0])
    return predictions

# 集成預測（簡單平均）
def stacking_predict(models, img):
    preds = predict_model(models, img)
    avg = np.mean(preds)
    return "Deepfake" if avg > 0.5 else "Real"

# 顯示預測結果
def show_prediction(img, models):
    label = stacking_predict(models, img)
    st.image(img, caption="輸入圖像", use_container_width=True)
    st.subheader(f"預測結果：**{label}**")

# UI Tab
models = load_models()
tab1, tab2 = st.tabs(["🖼️ 圖像偵測", "🎥 影片偵測"])

with tab1:
    st.header("上傳圖像進行 Deepfake 偵測")
    uploaded_image = st.file_uploader("選擇一張圖像", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="原始圖像", use_container_width=True)

        face_img = extract_face(pil_img)
        if face_img:
            st.image(face_img, caption="偵測到人臉", width=300)
            show_prediction(face_img, models)
        else:
            st.info("⚠️ 沒偵測到人臉，使用整張圖像預測")
            show_prediction(pil_img, models)

with tab2:
    st.header("影片偵測（處理前幾幀）")
    uploaded_video = st.file_uploader("選擇一段影片", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        st.info("🎬 正在分析影片...（取前 10 幀）")
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        shown = False
        max_frames = 10

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 3 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb)
                face_img = extract_face(pil_frame)
                if face_img:
                    st.image(face_img, caption=f"第 {frame_idx} 幀人臉", width=300)
                    label = stacking_predict(models, face_img)
                    st.subheader(f"預測結果：**{label}**")
                    shown = True
                    break
            frame_idx += 1

        cap.release()
        if not shown:
            st.warning("⚠️ 沒有偵測到人臉。")
