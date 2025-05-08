import numpy as np
import streamlit as st
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense
from PIL import Image
from mtcnn import MTCNN
import requests
import os
import cv2
import tempfile

# 🔽 下載模型（如果模型未下載過）
def download_model():
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    model_filename = "deepfake_cnn_model.h5"
    
    if not os.path.exists(model_filename):
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_filename, "wb") as f:
                f.write(response.content)
            print("✅ 模型檔案已成功下載")
        else:
            print(f"❌ 模型下載失敗，狀態碼：{response.status_code}")
            return None
    return model_filename

# 🔹 載入 ResNet50 模型
try:
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    resnet_classifier = Sequential([
        resnet_model,
        Dense(1, activation='sigmoid')  
    ])
    resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("✅ ResNet50 模型已成功載入")
except Exception as e:
    print(f"❌ 載入 ResNet50 模型錯誤：{e}")
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

# 🔹 中心裁切
def center_crop(img, target_size=(224, 224)):
    width, height = img.size
    new_width, new_height = target_size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    return img.crop((left, top, right, bottom))

# 🔹 圖片預處理
def preprocess_for_resnet(img):
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img = center_crop(img, (224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    resnet_input = np.expand_dims(img_array, axis=0)
    return resnet_input

# 🔹 預測
def predict_with_resnet(img):
    resnet_input = preprocess_for_resnet(img)
    prediction = resnet_classifier.predict(resnet_input)[0][0]
    label = "Deepfake" if prediction > 0.5 else "Real"
    return label, prediction

# 🔹 顯示預測結果
def show_prediction(img):
    label, confidence = predict_with_resnet(img)
    st.image(img, caption="輸入圖片", use_container_width=True)
    st.subheader(f"🔍 預測結果：**{label}**（信心值：{confidence:.2%}）")

# 🔹 Streamlit UI
st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖片與影片偵測器")

tab1, tab2 = st.tabs(["🖼️ 圖片偵測", "🎥 影片偵測"])

# ---------- 圖片 ----------
with tab1:
    st.header("圖片偵測")
    uploaded_image = st.file_uploader("請上傳圖片（jpg/png）", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="原始圖片", use_container_width=True)

        face_img = extract_face(pil_img)
        if face_img:
            st.image(face_img, caption="偵測到的人臉", use_container_width=False, width=300)
            show_prediction(face_img)
        else:
            st.info("⚠️ 未偵測到人臉，使用整張圖片進行分析")
            show_prediction(pil_img)

# ---------- 影片 ----------
with tab2:
    st.header("影片偵測（每 10 幀分析一次）")
    uploaded_video = st.file_uploader("請上傳影片（mp4/mov/avi）", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        st.info("🎬 擷取影片中... 請稍候")

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 10 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)

                face_img = extract_face(frame_pil)
                if face_img:
                    st.image(face_img, caption=f"第 {frame_idx} 幀偵測到人臉", use_container_width=False, width=300)
                    show_prediction(face_img)
                else:
                    st.image(frame_pil, caption=f"第 {frame_idx} 幀（未偵測到人臉）", use_container_width=True)
                    show_prediction(frame_pil)

            frame_idx += 1

        cap.release()
