import os
import numpy as np
import cv2
import tempfile
import requests
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 載入 ResNet50 模型
@st.cache_resource
def load_resnet_model():
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
    model = Sequential([
        base_model,
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

resnet_model = load_resnet_model()

# 載入 Custom CNN 模型
@st.cache_resource
def load_custom_model():
    model_path = os.path.join(tempfile.gettempdir(), "deepfake_cnn_model.h5")
    if not os.path.exists(model_path):
        url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
    return load_model(model_path)

custom_model = load_custom_model()

# 圖像預處理（不額外處理）
def preprocess_image(img):
    resized = cv2.resize(img, (256, 256))
    resnet_input = preprocess_input(np.expand_dims(resized.astype(np.float32), axis=0))
    custom_input = np.expand_dims(resized / 255.0, axis=0)
    return resized, resnet_input, custom_input

# 圖片處理
def process_image(file_bytes):
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    display_img, resnet_input, custom_input = preprocess_image(img)

    resnet_pred = resnet_model.predict(resnet_input)[0][0]
    resnet_label = "Deepfake" if resnet_pred > 0.5 else "Real"
    resnet_conf = resnet_pred if resnet_pred > 0.5 else 1 - resnet_pred

    custom_pred = custom_model.predict(custom_input)[0][0]
    custom_label = "Deepfake" if custom_pred > 0.5 else "Real"
    custom_conf = custom_pred if custom_pred > 0.5 else 1 - custom_pred

    rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    st.image(rgb_img, caption=f"🔍 ResNet50: {resnet_label} ({resnet_conf:.2%}) | Custom CNN: {custom_label} ({custom_conf:.2%})", use_container_width=True)

# 影片處理（每 10 幀抽樣）
def process_video(video_file):
    temp_path = os.path.join(tempfile.gettempdir(), "input_video.mp4")
    with open(temp_path, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(temp_path)
    frame_count = 0
    resnet_preds = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 10 == 0:
            frame_resized, resnet_input, _ = preprocess_image(frame)
            pred = resnet_model.predict(resnet_input)[0][0]
            label = "Deepfake" if pred > 0.5 else "Real"
            conf = pred if pred > 0.5 else 1 - pred
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            st.image(rgb_frame, caption=f"🎞️ Frame {frame_count}: {label} ({conf:.2%})", use_container_width=True)
            resnet_preds.append(pred)
    cap.release()

    if resnet_preds:
        st.line_chart(resnet_preds)

# Streamlit App UI
st.set_page_config(page_title="Deepfake 偵測", layout="centered")
st.title("🧠 Deepfake 偵測系統")
file_type = st.radio("選擇上傳檔案類型：", ("圖片", "影片"))
uploaded_file = st.file_uploader("請上傳圖片（jpg/png）或影片（mp4）", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    try:
        if file_type == "圖片" and uploaded_file.type.startswith("image"):
            bytes_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            process_image(bytes_data)
        elif file_type == "影片" and uploaded_file.type.startswith("video"):
            st.info("處理影片中，請稍後...")
            process_video(uploaded_file)
        else:
            st.warning("檔案類型與選擇不符，請重新上傳。")
    except Exception as e:
        st.error(f"❌ 發生錯誤：{e}")
