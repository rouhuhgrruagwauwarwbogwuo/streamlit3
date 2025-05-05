import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.layers import Dense
import requests
from mtcnn import MTCNN

# ⚙️ 設定頁面
st.set_page_config(page_title="Deepfake Detector", layout="centered")

# 📦 載入模型
@st.cache_resource
def load_models():
    # 載入自訂 CNN 模型
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    response = requests.get(model_url)
    model_path = '/tmp/deepfake_cnn_model.h5'
    with open(model_path, 'wb') as f:
        f.write(response.content)
    custom_model = load_model(model_path)

    # 載入 ResNet50 模型
    resnet_base = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
    resnet_classifier = Sequential([
        resnet_base,
        Dense(1, activation='sigmoid')
    ])
    resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return custom_model, resnet_classifier

custom_model, resnet_classifier = load_models()

# 🔍 圖像預處理
def preprocess_for_both_models(image_array):
    img_resized = cv2.resize(image_array, (256, 256))
    resnet_input = preprocess_input(np.expand_dims(img_resized, axis=0))
    custom_input = np.expand_dims(img_resized / 255.0, axis=0)
    return resnet_input, custom_input

# 🧠 預測函數
def predict_with_models(image_array):
    resnet_input, custom_input = preprocess_for_both_models(image_array)
    resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
    custom_pred = custom_model.predict(custom_input)[0][0]
    resnet_label = "Deepfake" if resnet_pred > 0.5 else "Real"
    custom_label = "Deepfake" if custom_pred > 0.5 else "Real"
    return resnet_label, resnet_pred, custom_label, custom_pred

# 👤 人臉擷取
def extract_face(img):
    detector = MTCNN()
    faces = detector.detect_faces(img)
    if faces:
        x, y, w, h = faces[0]['box']
        return img[y:y+h, x:x+w]
    return img

# 🎞️ 影片處理
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx > 100:  # 最多處理前 100 幀
            break

        if frame_idx % 10 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_img = extract_face(rgb_frame)
            resnet_label, resnet_conf, custom_label, custom_conf = predict_with_models(face_img)

            annotated = cv2.putText(
                rgb_frame.copy(),
                f"ResNet50: {resnet_label} ({resnet_conf:.2%}) | Custom: {custom_label} ({custom_conf:.2%})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            frames.append(annotated)

        frame_idx += 1

    cap.release()
    return frames

# 🖼️ 主介面
st.title("🕵️ Deepfake 圖片與影片偵測器")

tab1, tab2 = st.tabs(["📷 上傳圖片", "🎥 上傳影片"])

with tab1:
    uploaded_image = st.file_uploader("請選擇一張圖片", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        img_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_img = extract_face(img_rgb)
        resnet_label, resnet_conf, custom_label, custom_conf = predict_with_models(face_img)

        st.image(img_rgb, caption="原始圖片", use_container_width=True)
        st.markdown(f"""
        **ResNet50 預測**: {resnet_label} ({resnet_conf:.2%})  
        **Custom CNN 預測**: {custom_label} ({custom_conf:.2%})
        """)

with tab2:
    uploaded_video = st.file_uploader("請選擇影片檔", type=["mp4", "mov", "avi"])
    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
            tmp_vid.write(uploaded_video.read())
            video_path = tmp_vid.name

        st.info("影片處理中，請稍候...")
        frames = process_video(video_path)
        for f in frames:
            st.image(f, use_container_width=True)

# 🧪 擴充功能（高通濾波、FFT、YCbCr）未直接整合，但可以根據需要插入
