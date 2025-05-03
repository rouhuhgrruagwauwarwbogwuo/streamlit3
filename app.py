import os
import numpy as np
import cv2
import tempfile
import requests
import h5py
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense

# 模型下載網址
MODEL_URL = "https://huggingface.co/wuwuwu123123/deepfake/resolve/main/deepfake_cnn_model.h5"

@st.cache_resource
def download_model():
    model_path = os.path.join(tempfile.gettempdir(), "deepfake_cnn_model.h5")
    if not os.path.exists(model_path):
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
        else:
            st.error("❌ 模型下載失敗")
            raise Exception("模型下載失敗")
    return load_model(model_path)

try:
    custom_model = download_model()
except Exception as e:
    st.error(f"❌ 模型載入失敗: {e}")
    st.stop()

resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
resnet_classifier = Sequential([
    resnet_model,
    Dense(1, activation='sigmoid')
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 預處理與強化

def enhance_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharp = cv2.filter2D(img, -1, kernel)
    return img_sharp

def preprocess_for_models(img):
    img = enhance_image(img)
    img_resized = cv2.resize(img, (256, 256))
    resnet_input = preprocess_input(np.expand_dims(img_resized, axis=0))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    clahe_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    custom_input = np.expand_dims(clahe_rgb / 255.0, axis=0)
    return resnet_input, custom_input, img_resized

# 人臉偵測

def detect_face_region(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return img[y:y+h, x:x+w]

# 視覺化

def plot_confidence(resnet_conf, custom_conf, combined_conf):
    fig, ax = plt.subplots()
    models = ['ResNet50', 'Custom CNN', 'Combined']
    confs = [resnet_conf, custom_conf, combined_conf]
    ax.bar(models, confs, color=['blue', 'green', 'purple'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Confidence')
    st.pyplot(fig)

# 移動平均

def smooth_predictions(pred_list, window_size=5):
    if len(pred_list) < window_size:
        return pred_list
    return np.convolve(pred_list, np.ones(window_size)/window_size, mode='valid')

# 圖片處理

def process_image(file_bytes):
    try:
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        face = detect_face_region(img)
        if face is None:
            st.warning("⚠️ 未偵測到人臉")
            return
        resnet_input, custom_input, display_img = preprocess_for_models(face)
        resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
        custom_pred = custom_model.predict(custom_input)[0][0]
        combined_pred = (resnet_pred + custom_pred) / 2
        label = "Deepfake" if combined_pred > 0.5 else "Real"
        confidence = combined_pred if combined_pred > 0.5 else 1 - combined_pred
        st.image(img, caption=f"預測結果：{label} ({confidence:.2%})", use_container_width=True)
        plot_confidence(resnet_pred, custom_pred, combined_pred)
    except Exception as e:
        st.error(f"❌ 圖片處理錯誤: {e}")

# 影片處理

def process_video_and_generate_result(video_file):
    try:
        temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            st.error("❌ 無法打開影片")
            return
        frame_preds = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 10 == 0:
                face = detect_face_region(frame)
                if face is None:
                    continue
                resnet_input, custom_input, display_img = preprocess_for_models(face)
                resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
                custom_pred = custom_model.predict(custom_input)[0][0]
                combined_pred = (resnet_pred + custom_pred) / 2
                label = "Deepfake" if combined_pred > 0.5 else "Real"
                confidence = combined_pred if combined_pred > 0.5 else 1 - combined_pred
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(display_img, f"{label} ({confidence:.2%})", (10, 30), font, 1, (0, 255, 0), 2)
                st.image(display_img, caption=f"幀 {frame_count}: {label} ({confidence:.2%})", use_container_width=True)
                frame_preds.append(combined_pred)
        cap.release()
        smoothed = smooth_predictions(frame_preds)
        st.line_chart(smoothed)
        st.success("🎉 偵測完成！")
    except Exception as e:
        st.error(f"❌ 影片處理錯誤: {e}")

# Streamlit UI

st.title("🕵️ Deepfake 偵測 App (人臉強化版)")
option = st.radio("請選擇檔案類型：", ("圖片", "影片"))

uploaded_file = st.file_uploader("📤 上傳檔案", type=["jpg", "jpeg", "png", "mp4", "mov"])

if uploaded_file is not None:
    try:
        if option == "圖片" and uploaded_file.type.startswith("image"):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            process_image(file_bytes)
        elif option == "影片" and uploaded_file.type.startswith("video"):
            st.markdown("### 處理影片中...")
            process_video_and_generate_result(uploaded_file)
        else:
            st.warning("請確認上傳的檔案類型與選擇一致。")
    except Exception as e:
        st.error(f"❌ 發生錯誤: {e}")
