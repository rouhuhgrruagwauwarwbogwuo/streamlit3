import streamlit as st
import numpy as np
import cv2
import tempfile
import requests
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 下載模型（只會下載一次）
@st.cache_resource
def download_model():
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    model_path = "deepfake_cnn_model.h5"
    if not os.path.exists(model_path):
        with st.spinner("下載模型中..."):
            r = requests.get(model_url)
            with open(model_path, "wb") as f:
                f.write(r.content)
    return load_model(model_path)

# 使用 ResNet50 模型進行預測
def preprocess_for_both_models(image_path):
    img = image.load_img(image_path, target_size=(256, 256))  # 調整大小
    img_array = image.img_to_array(img)
    
    # ResNet50 需要特別的 preprocess_input
    resnet_input = preprocess_input(np.expand_dims(img_array, axis=0))
    
    # 自訂 CNN 只需要正規化 (0~1)
    custom_input = np.expand_dims(img_array / 255.0, axis=0)
    
    return resnet_input, custom_input

# 進行 ResNet50 預測
def predict_with_both_models(image_path):
    resnet_input, custom_input = preprocess_for_both_models(image_path)
    
    # ResNet50 預測
    resnet_prediction = resnet_classifier.predict(resnet_input)[0][0]
    resnet_label = "Deepfake" if resnet_prediction > 0.5 else "Real"
    
    # 自訂 CNN 模型預測
    custom_prediction = custom_model.predict(custom_input)[0][0]
    custom_label = "Deepfake" if custom_prediction > 0.5 else "Real"
    
    return resnet_label, resnet_prediction, custom_label, custom_prediction

# 人臉偵測
def detect_face(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            img_array = cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_img = Image.fromarray(img_array)
        return face_img
    return img

# 顯示圖片和預測結果
def show_prediction(image_path):
    resnet_label, resnet_confidence, custom_label, custom_confidence = predict_with_both_models(image_path)
    
    # 顯示圖片
    img = image.load_img(image_path, target_size=(256, 256))
    st.image(img, caption="原始圖片", use_container_width=True)
    
    # 顯示預測結果
    st.subheader(f"✅ 預測結果：")
    st.write(f"ResNet50 預測：{resnet_label} ({resnet_confidence:.2%} 信心分數)")
    st.write(f"Custom CNN 預測：{custom_label} ({custom_confidence:.2%} 信心分數)")
    
    # 顯示偵測到的人臉
    face_img = detect_face(img)
    st.image(face_img, caption="偵測到的人臉", use_container_width=True)

# 主頁面設定
st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖片與影片偵測器")

model = download_model()

# ---------- 圖片偵測 ----------
st.header("圖片偵測")
uploaded_image = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])
if uploaded_image:
    pil_img = Image.open(uploaded_image).convert("RGB")
    show_prediction(uploaded_image)  # 顯示圖片與預測結果

# ---------- 影片偵測 ----------
st.header("影片偵測（每 10 幀抽圖）")
uploaded_video = st.file_uploader("上傳影片", type=["mp4", "mov", "avi"])
if uploaded_video:
    st.video(uploaded_video)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

    st.info("🎬 擷取影片幀與進行預測中...")
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 10 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            st.image(pil_img, caption=f"第 {frame_idx} 幀", use_container_width=True)
            show_prediction(pil_img)  # 顯示每一幀的預測結果
        frame_idx += 1
    cap.release()
