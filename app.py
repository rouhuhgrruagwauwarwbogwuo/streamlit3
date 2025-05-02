import streamlit as st
import numpy as np
import cv2
import requests
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---------- 模型下載與載入 ----------
MODEL_PATH = "deepfake_cnn_model.h5"
MODEL_URL = "https://huggingface.co/wuwuwu123123/deepfake3/resolve/main/deepfake_cnn_model.h5"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return load_model(MODEL_PATH)

model = download_and_load_model()

# ---------- 圖片預處理 ----------
def preprocess_image(uploaded_file, target_size=(256, 256)):
    try:
        img = Image.open(uploaded_file).convert("RGB").resize(target_size)
        img_array = np.array(img)

        # 中值濾波去噪
        img_array = cv2.medianBlur(img_array.astype('uint8'), 3)

        # CLAHE 增強對比度
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)
        img_array = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        # 正規化
        img_array = img_array / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"預處理錯誤：{e}")
        return None

# ---------- 預測結果 ----------
def predict_image(uploaded_file):
    img_array = preprocess_image(uploaded_file)

    if img_array is not None:
        prediction = model.predict(img_array)[0][0]
        label = "Deepfake" if prediction > 0.5 else "Real"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        return label, confidence
    else:
        return None, None

# ---------- 顯示圖片預測 ----------
def show_image_prediction(uploaded_file):
    label, confidence = predict_image(uploaded_file)
    
    if label is not None:
        st.image(uploaded_file, caption="上傳的圖片", use_column_width=True)
        st.markdown(f"### 🔍 預測結果：**{label}**")
        st.markdown(f"### 📊 信心分數：**{confidence:.2%}**")
    else:
        st.error("無法顯示預測結果")

# ---------- 影片預測 ----------
def predict_video(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_interval = 5
    frame_count = 0
    frame_buffer = []
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break  # 影片讀取結束

        if frame_count % frame_interval == 0:
            # 偵測每幀中的人臉
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (256, 256))
                face_array = np.expand_dims(face_resized / 255.0, axis=0)
                prediction = model.predict(face_array)[0][0]
                predictions.append(prediction)

        frame_count += 1

    cap.release()

    if predictions:
        avg_prediction = np.mean(predictions)
        label = "Deepfake" if avg_prediction > 0.5 else "Real"
        confidence = avg_prediction if avg_prediction > 0.5 else 1 - avg_prediction
        return label, confidence
    else:
        return None, None

# ---------- 顯示影片預測 ----------
def show_video_prediction(uploaded_video):
    label, confidence = predict_video(uploaded_video)
    
    if label is not None:
        st.video(uploaded_video)
        st.markdown(f"### 🔍 預測結果：**{label}**")
        st.markdown(f"### 📊 信心分數：**{confidence:.2%}**")
    else:
        st.error("無法顯示預測結果")

# ---------- Streamlit UI ----------
st.title("🧠 Deepfake 偵測 App")
st.write("請選擇圖片或影片，系統將自動判斷是否為 Deepfake。")

choice = st.radio("選擇檔案類型", ("圖片", "影片"))

if choice == "圖片":
    uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        show_image_prediction(uploaded_file)

elif choice == "影片":
    uploaded_video = st.file_uploader("上傳影片", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        show_video_prediction(uploaded_video)
