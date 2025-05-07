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
from moviepy.editor import ImageSequenceClip

# ✅ 檢查並串流下載模型檔案

def download_model():
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    model_filename = "deepfake_cnn_model.h5"

    if not os.path.exists(model_filename):
        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            with open(model_filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ 模型檔案下載成功")
        else:
            print(f"❌ 下載失敗，狀態碼：{response.status_code}")
            return None
    else:
        print(f"📁 模型檔案 {model_filename} 已存在")
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

# 🔹 初始化 MTCNN 人臉檢測器
detector = MTCNN()

# 🔹 高通濾波增強
def high_pass_filter(img_array):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filtered_img = cv2.filter2D(img_array, -1, kernel)
    return filtered_img

# 🔹 CLAHE + 銳化預處理
def preprocess_image(img, target_size=(256, 256)):
    img_array = np.array(img.resize(target_size)).astype('uint8')
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    img_rgb = img_rgb / 255.0
    img_rgb = high_pass_filter(img_rgb)
    return np.expand_dims(img_rgb, axis=0)

# 🔹 人臉擷取
def extract_face(pil_img):
    img_rgb = np.array(pil_img)
    faces = detector.detect_faces(img_rgb)
    if faces:
        x, y, w, h = faces[0]['box']
        face = img_rgb[y:y+h, x:x+w]
        return Image.fromarray(face)
    return pil_img  # 若無人臉則返回整張圖

# 🔹 預測
def predict_labels(pil_img):
    resized_img = pil_img.resize((256, 256))
    resnet_input = preprocess_input(np.expand_dims(np.array(resized_img), axis=0))
    cnn_input = preprocess_image(pil_img)
    resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
    cnn_pred = custom_model.predict(cnn_input)[0][0] if custom_model else 0
    return resnet_pred, cnn_pred

# 🔹 Streamlit App
st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖片與影片偵測器")
tab1, tab2 = st.tabs(["🖼️ 圖片偵測", "🎥 影片偵測"])

with tab1:
    st.header("圖片偵測")
    uploaded_image = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        face_img = extract_face(pil_img)
        st.image(face_img.resize((200, 200)), caption="人臉或原圖 (縮小版)", use_container_width=False)
        r_score, c_score = predict_labels(face_img)
        st.subheader(f"ResNet50 預測: {'Deepfake' if r_score > 0.5 else 'Real'} ({r_score:.2%})")
        st.subheader(f"Custom CNN 預測: {'Deepfake' if c_score > 0.5 else 'Real'} ({c_score:.2%})")

with tab2:
    st.header("影片偵測")
    uploaded_video = st.file_uploader("上傳影片", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        frames = []
        pred_labels = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            face_img = extract_face(pil_frame)

            r_score, c_score = predict_labels(face_img)
            label = "Deepfake" if r_score > 0.5 else "Real"
            color = (0, 0, 255) if label == "Deepfake" else (0, 255, 0)

            annotated = cv2.putText(np.array(face_img.resize((256, 256))),
                                    f"{label} {r_score:.2%}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, color, 2, cv2.LINE_AA)
            frames.append(annotated)

        cap.release()

        # 輸出影片
        out_path = "output.mp4"
        clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames], fps=10)
        clip.write_videofile(out_path, codec="libx264")
        st.video(out_path)
        st.success("影片處理完成並已標記預測結果！")
