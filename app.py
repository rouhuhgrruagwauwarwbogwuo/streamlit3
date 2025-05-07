import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from PIL import Image
from mtcnn import MTCNN
import tempfile
import os
import requests

# 🔽 下載 ResNet50 模型（假設已經加載）
try:
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    resnet_classifier = Sequential([
        resnet_model,
        Dense(1, activation='sigmoid')
    ])
    resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("ResNet50 模型已成功載入")
except Exception as e:
    print(f"載入 ResNet50 模型時發生錯誤：{e}")
    resnet_classifier = None

# 🔹 初始化 MTCNN 人臉檢測器
detector = MTCNN()

# 🔹 擷取圖片中的人臉
def extract_face(pil_img):
    img_array = np.array(pil_img)
    faces = detector.detect_faces(img_array)

    if len(faces) > 0:
        x, y, width, height = faces[0]['box']
        face = img_array[y:y+height, x:x+width]
        face_pil = Image.fromarray(face)
        return face_pil
    else:
        return None

# 🔹 中心裁切函數
def center_crop(img, target_size=(224, 224)):
    width, height = img.size
    new_width, new_height = target_size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    return img.crop((left, top, right, bottom))

# 🔹 圖片預處理（包含 CLAHE 增強）
def preprocess_for_resnet(img):
    img = img.resize((256, 256), Image.Resampling.LANCZOS)  # 重設大小
    img = center_crop(img, (224, 224))  # 中心裁切
    img_array = np.array(img)
    img_array = img_array.astype(np.float32) / 255.0  # 標準化
    img_array = preprocess_input(img_array)  # ResNet50 預處理
    return np.expand_dims(img_array, axis=0)  # 擴展維度以符合模型要求

# 🔹 模型預測
def predict_with_resnet(img):
    img_input = preprocess_for_resnet(img)
    prediction = resnet_classifier.predict(img_input)[0][0]
    label = "Deepfake" if prediction > 0.5 else "Real"
    return label, prediction

# 🔹 顯示預測結果
def show_prediction(img):
    label, confidence = predict_with_resnet(img)
    st.image(img, caption="原始圖片", use_container_width=True)
    st.subheader(f"ResNet50: {label} ({confidence:.2%})")

# 🔹 Streamlit 主頁面
st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖片與影片偵測器")

# 側邊欄選項
only_resnet = st.sidebar.checkbox("僅顯示 ResNet50 預測", value=True)

# 分頁
tab1, tab2 = st.tabs(["🖼️ 圖片偵測", "🎥 影片偵測"])

# ---------- 圖片 ----------
with tab1:
    st.header("圖片偵測")
    uploaded_image = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="原始圖片", use_container_width=True)

        face_img = extract_face(pil_img)
        if face_img:
            st.image(face_img, caption="偵測到的人臉", use_container_width=False, width=300)
            show_prediction(face_img)
        else:
            st.write("未偵測到人臉，使用整體圖片進行預測")
            show_prediction(pil_img)

# ---------- 影片 ----------
with tab2:
    st.header("影片偵測（僅分析前幾幀）")
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
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                face_img = extract_face(frame_pil)
                if face_img:
                    st.image(face_img, caption="偵測到的人臉", use_container_width=False, width=300)
                    show_prediction(face_img)
                    break
            frame_idx += 1
        cap.release()
