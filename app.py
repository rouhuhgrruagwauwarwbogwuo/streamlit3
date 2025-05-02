import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tempfile
import os
import matplotlib.pyplot as plt

# 載入模型
model = load_model('deepfake_cnn_model.h5')

# ==== 圖片預處理函數 ====
def preprocess_image(img_path, target_size=(256, 256)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)

        img_array = cv2.medianBlur(img_array.astype('uint8'), 3)

        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)
        img_array = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        img_array = img_array / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"圖片預處理錯誤: {e}")
        return None

# ==== 圖片預測 ====
def predict_image(model, img_path):
    img_array = preprocess_image(img_path)
    if img_array is None:
        return None, None
    prediction = model.predict(img_array)[0][0]
    label = "Deepfake" if prediction > 0.5 else "Real"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

# ==== 影片處理預設函數 ====
def preprocess_face(face):
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_equalized = cv2.equalizeHist(face_gray)
    face_normalized = cv2.cvtColor(face_equalized, cv2.COLOR_GRAY2RGB)
    face_blurred = cv2.GaussianBlur(face_normalized, (3, 3), 0)
    return face_blurred

def preprocess_for_model(face):
    face_resized = cv2.resize(face, (256, 256))
    img_array = image.img_to_array(face_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def smooth_predictions(predictions, window_size=5):
    smoothed_predictions = []
    for i in range(len(predictions)):
        start = max(0, i - window_size // 2)
        end = min(len(predictions), i + window_size // 2 + 1)
        smoothed_predictions.append(np.mean(predictions[start:end]))
    return smoothed_predictions

# ==== Streamlit App ====
st.title("🧠 Deepfake 圖片與影片分析 App")

mode = st.radio("請選擇分析模式", ["圖片", "影片"])

if mode == "圖片":
    uploaded_img = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img:
            temp_img.write(uploaded_img.read())
            temp_img_path = temp_img.name

        label, confidence = predict_image(model, temp_img_path)

        if label:
            st.image(temp_img_path, caption=f"預測結果：{label} ({confidence:.2%} 信心分數)", use_column_width=True)
        else:
            st.error("圖片預測失敗，請確認圖片是否有效")

elif mode == "影片":
    uploaded_video = st.file_uploader("上傳影片（mp4）", type=["mp4"])
    frame_interval = st.slider("分析幀的間隔數", 1, 30, 5)

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        frame_count = 0
        frame_buffer = []

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    face_processed = preprocess_face(face)
                    face_input = preprocess_for_model(face_processed)
                    prediction = model.predict(face_input)[0][0]

                    frame_buffer.append(prediction)

                    label = "Deepfake" if prediction > 0.5 else "Real"
                    color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{label} ({prediction:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if len(frame_buffer) > 5:
                smoothed = smooth_predictions(frame_buffer)
                frame_buffer.clear()
                avg_pred = np.mean(smoothed)
                label = "Deepfake" if avg_pred > 0.5 else "Real"
                color = (0, 0, 255) if avg_pred > 0.5 else (0, 255, 0)
                cv2.putText(frame, f"Smoothed: {label} ({avg_pred:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(rgb_frame, channels="RGB", use_column_width=True)

            frame_count += 1

        cap.release()
        os.remove(video_path)
