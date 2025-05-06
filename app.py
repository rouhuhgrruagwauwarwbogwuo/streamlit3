import streamlit as st
import numpy as np
import cv2
import os
import requests
from mtcnn import MTCNN
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense

# 頁面設定
st.set_page_config(page_title="Deepfake 偵測", layout="wide")

# 載入模型
@st.cache_resource
def load_models():
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    response = requests.get(model_url)
    model_path = '/tmp/deepfake_cnn_model.h5'
    with open(model_path, 'wb') as f:
        f.write(response.content)
    custom_model = load_model(model_path)
    
    resnet_base = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
    resnet_classifier = Sequential([resnet_base, Dense(1, activation='sigmoid')])
    resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return custom_model, resnet_classifier

custom_model, resnet_classifier = load_models()
detector = MTCNN()

# 人臉擷取
def extract_face(image_np):
    faces = detector.detect_faces(image_np)
    if faces:
        x, y, w, h = faces[0]['box']
        return image_np[y:y+h, x:x+w]
    return image_np

# 預處理
def preprocess_for_both_models(img_array):
    img_array = cv2.resize(img_array, (256, 256))
    resnet_input = preprocess_input(np.expand_dims(img_array.copy(), axis=0))
    custom_input = np.expand_dims(img_array / 255.0, axis=0)
    return resnet_input, custom_input

# 預測
def predict(img_array):
    resnet_input, custom_input = preprocess_for_both_models(img_array)
    resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
    custom_pred = custom_model.predict(custom_input)[0][0]
    resnet_label = "Deepfake" if resnet_pred > 0.5 else "Real"
    custom_label = "Deepfake" if custom_pred > 0.5 else "Real"
    return resnet_label, resnet_pred, custom_label, custom_pred

# 圖片預測
def handle_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(img)
    face = extract_face(img_np)
    resnet_label, resnet_conf, custom_label, custom_conf = predict(face)
    st.image(img, caption="上傳的圖片", use_container_width=True)
    st.markdown(f"**ResNet50** 預測結果：`{resnet_label}` ({resnet_conf:.2%})")
    st.markdown(f"**Custom CNN** 預測結果：`{custom_label}` ({custom_conf:.2%})")

# 影片預測（每10幀顯示一張圖）
def handle_video(uploaded_file):
    video_path = f"/tmp/{uploaded_file.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    st_frame = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 10 != 0:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = extract_face(rgb_frame)
        resnet_label, resnet_conf, custom_label, custom_conf = predict(face)

        # 繪製標籤
        label = f"ResNet: {resnet_label} ({resnet_conf:.2%}) | CNN: {custom_label} ({custom_conf:.2%})"
        cv2.putText(rgb_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        st_frame.image(rgb_frame, caption=f"第 {frame_count} 幀", channels="RGB", use_column_width=True)

    cap.release()

# Streamlit 介面
st.title("🧠 Deepfake 圖片與影片偵測")

tab1, tab2 = st.tabs(["🖼️ 上傳圖片", "🎬 上傳影片"])

with tab1:
    image_file = st.file_uploader("請上傳圖片 (jpg, png)", type=["jpg", "jpeg", "png"])
    if image_file:
        handle_image(image_file)

with tab2:
    video_file = st.file_uploader("請上傳影片 (mp4, mov, avi)", type=["mp4", "mov", "avi"])
    if video_file:
        handle_video(video_file)
