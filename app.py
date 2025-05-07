import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from PIL import Image
from mtcnn import MTCNN
import tempfile
import os
import requests
import h5py

# 檢查並下載模型檔案
def download_model():
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    model_filename = "deepfake_cnn_model.h5"
    
    if not os.path.exists(model_filename):
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_filename, "wb") as f:
                f.write(response.content)
    return model_filename

# 🔹 載入 ResNet50 模型
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
resnet_classifier = Sequential([
    resnet_model,
    Dense(1, activation='sigmoid')
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔹 載入自訂 CNN 模型
model_path = download_model()
custom_model = load_model(model_path) if model_path else None

# 🔹 初始化 MTCNN
detector = MTCNN()

def high_pass_filter(img_array):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return cv2.filter2D(img_array, -1, kernel)

def preprocess_image(image_path, target_size=(256, 256)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img).astype('uint8')
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_gray = clahe.apply(img_gray)
    img_array = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    img_array = img_array / 255.0
    return np.expand_dims(high_pass_filter(img_array), axis=0)

def extract_face(img):
    img_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    if faces:
        x, y, w, h = faces[0]['box']
        face = img_rgb[y:y+h, x:x+w]
        return Image.fromarray(face)
    return None

def preprocess_for_both_models(img):
    img = img.resize((256, 256))
    img_array = np.array(img)
    resnet_input = preprocess_input(np.expand_dims(img_array, axis=0))
    custom_input = np.expand_dims(img_array / 255.0, axis=0)
    return resnet_input, custom_input

def predict_with_both_models(img):
    resnet_input, custom_input = preprocess_for_both_models(img)
    resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
    resnet_label = "Deepfake" if resnet_pred > 0.5 else "Real"
    custom_pred = custom_model.predict(custom_input)[0][0] if custom_model else 0
    custom_label = "Deepfake" if custom_pred > 0.5 else "Real"
    return resnet_label, resnet_pred, custom_label, custom_pred

def show_prediction(img):
    resnet_label, resnet_conf, custom_label, custom_conf = predict_with_both_models(img)
    st.image(img, caption="原始圖片", width=400)
    st.subheader(f"ResNet50: {resnet_label} ({resnet_conf:.2%})\nCustom CNN: {custom_label} ({custom_conf:.2%})")

st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖片與影片偵測器")

tab1, tab2 = st.tabs(["🖼️ 圖片偵測", "🎥 影片偵測"])

with tab1:
    st.header("圖片偵測")
    uploaded_image = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        face_img = extract_face(pil_img)
        show_prediction(face_img if face_img else pil_img)

with tab2:
    st.header("影片偵測（逐幀標註輸出影片）")
    uploaded_video = st.file_uploader("上傳影片", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
            tmp_input.write(uploaded_video.read())
            input_path = tmp_input.name

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        st.info("🎬 處理影片中...")

        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face_img = extract_face(frame_pil)
            if face_img:
                resnet_label, resnet_conf, custom_label, custom_conf = predict_with_both_models(face_img)
                label_text = f"ResNet: {resnet_label} ({resnet_conf:.0%}) | Custom: {custom_label} ({custom_conf:.0%})"
                color = (0, 0, 255) if resnet_label == "Deepfake" else (0, 255, 0)
                cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else:
                cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            out.write(frame)

        cap.release()
        out.release()

        st.success("✅ 處理完成！以下是標註後的影片：")
        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button("⬇️ 下載標註後影片", f.read(), file_name="annotated_result.mp4")
