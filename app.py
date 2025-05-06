import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from mtcnn import MTCNN
import requests
from PIL import Image

st.set_page_config(page_title="Deepfake 偵測", layout="centered")

@st.cache_resource
def load_models():
    # 下載自訂 CNN 模型
    cnn_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    cnn_path = "/tmp/deepfake_cnn_model.h5"
    if not os.path.exists(cnn_path):
        r = requests.get(cnn_url)
        with open(cnn_path, "wb") as f:
            f.write(r.content)
    custom_model = load_model(cnn_path)

    # 下載 ResNet50 分類器（接 Dense 的那個）
    resnet_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/resnet50_classifier.h5"
    resnet_path = "/tmp/resnet50_classifier.h5"
    if not os.path.exists(resnet_path):
        r = requests.get(resnet_url)
        with open(resnet_path, "wb") as f:
            f.write(r.content)
    resnet_model = load_model(resnet_path)

    return custom_model, resnet_model

custom_model, resnet_model = load_models()

# 預處理圖片
def preprocess_img_for_models(img_array):
    img_resized = cv2.resize(img_array, (256, 256))
    rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    norm = rgb / 255.0
    resnet_input = preprocess_input(np.expand_dims(rgb.astype('float32'), axis=0))
    custom_input = np.expand_dims(norm, axis=0)
    return resnet_input, custom_input

# 人臉偵測
def detect_face(img):
    detector = MTCNN()
    faces = detector.detect_faces(img)
    if faces:
        x, y, w, h = faces[0]["box"]
        return img[y:y+h, x:x+w]
    return img

# 圖片預測
def predict_image(img_array):
    face = detect_face(img_array)
    resnet_input, custom_input = preprocess_img_for_models(face)
    resnet_pred = float(resnet_model.predict(resnet_input)[0][0])
    custom_pred = float(custom_model.predict(custom_input)[0][0])
    return resnet_pred, custom_pred

# 顯示圖片預測結果
def display_prediction(img, resnet_pred, custom_pred):
    st.image(img, caption="上傳圖片", use_container_width=True)
    st.markdown(f"🔍 **ResNet50 預測**：{'Deepfake' if resnet_pred > 0.5 else 'Real'} ({resnet_pred:.2%})")
    st.markdown(f"🧠 **自訂 CNN 預測**：{'Deepfake' if custom_pred > 0.5 else 'Real'} ({custom_pred:.2%})")

# 影片逐幀處理與顯示
def process_video_file(video_bytes):
    with open("/tmp/temp_video.mp4", "wb") as f:
        f.write(video_bytes.read())
    cap = cv2.VideoCapture("/tmp/temp_video.mp4")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx > 100:  # 限制最多顯示 100 幀
            break
        if frame_idx % 10 == 0:
            resnet_pred, custom_pred = predict_image(frame)
            label = f"ResNet: {'DF' if resnet_pred > 0.5 else 'Real'} ({resnet_pred:.1%}), CNN: {'DF' if custom_pred > 0.5 else 'Real'} ({custom_pred:.1%})"
            frame = cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"第 {frame_idx} 幀", use_column_width=True)
        frame_idx += 1
    cap.release()

# UI
st.title("🧠 Deepfake 圖片與影片偵測器")

# 圖片上傳
img_file = st.file_uploader("📸 上傳圖片", type=["jpg", "jpeg", "png"])
if img_file:
    img = Image.open(img_file).convert("RGB")
    img_array = np.array(img)
    resnet_pred, custom_pred = predict_image(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    display_prediction(img, resnet_pred, custom_pred)

# 影片上傳
video_file = st.file_uploader("🎥 上傳影片", type=["mp4", "mov", "avi"])
if video_file:
    st.info("影片分析中，請稍候...")
    process_video_file(video_file)
