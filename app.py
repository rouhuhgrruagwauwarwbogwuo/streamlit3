import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import requests
from io import BytesIO
from mtcnn import MTCNN
import tempfile

# ✅ 必須是第一個 Streamlit 指令
st.set_page_config(page_title="Deepfake 檢測", layout="centered")

st.title("🔍 Deepfake 檢測應用程式")

# 下載並載入自訂 CNN 模型
@st.cache_resource

def load_custom_model():
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    response = requests.get(model_url)
    model_path = '/tmp/deepfake_cnn_model.h5'
    with open(model_path, 'wb') as f:
        f.write(response.content)
    return load_model(model_path)

custom_model = load_custom_model()

# ResNet50 模型與分類器
resnet_base = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
resnet_classifier = Sequential([
    resnet_base,
    Dense(1, activation='sigmoid')
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 預處理圖片，供 ResNet50 與自訂 CNN 使用
def preprocess_for_both_models(img_array):
    img_array_resized = cv2.resize(img_array, (256, 256))
    resnet_input = preprocess_input(np.expand_dims(img_array_resized, axis=0))
    custom_input = np.expand_dims(img_array_resized / 255.0, axis=0)
    return resnet_input, custom_input

# MTCNN 擷取人臉
def extract_face(img):
    detector = MTCNN()
    faces = detector.detect_faces(img)
    if faces:
        x, y, w, h = faces[0]['box']
        return img[y:y+h, x:x+w]
    return None

# 高通濾波處理
def apply_highpass_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F)

# 預測函數
def predict_with_models(img_array):
    resnet_input, custom_input = preprocess_for_both_models(img_array)
    resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
    custom_pred = custom_model.predict(custom_input)[0][0]
    resnet_label = "Deepfake" if resnet_pred > 0.5 else "Real"
    custom_label = "Deepfake" if custom_pred > 0.5 else "Real"
    return resnet_label, resnet_pred, custom_label, custom_pred

# 顯示圖片與預測結果
def display_prediction(img_array, resnet_label, resnet_conf, custom_label, custom_conf):
    st.image(img_array, caption="上傳圖片", use_column_width=True)
    st.markdown(f"**ResNet50** 預測結果：{resnet_label} ({resnet_conf:.2%})")
    st.markdown(f"**Custom CNN** 預測結果：{custom_label} ({custom_conf:.2%})")

# 📷 上傳圖片
tab1, tab2 = st.tabs(["圖片預測", "影片分析"])

with tab1:
    uploaded_image = st.file_uploader("請上傳圖片：", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face = extract_face(img_rgb)
        if face is not None:
            face = cv2.resize(face, (256, 256))
            resnet_label, resnet_conf, custom_label, custom_conf = predict_with_models(face)
            display_prediction(face, resnet_label, resnet_conf, custom_label, custom_conf)
        else:
            img_resized = cv2.resize(img_rgb, (256, 256))
            resnet_label, resnet_conf, custom_label, custom_conf = predict_with_models(img_resized)
            display_prediction(img_resized, resnet_label, resnet_conf, custom_label, custom_conf)

with tab2:
    uploaded_video = st.file_uploader("請上傳影片：", type=["mp4", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(video_path)
        cap = cv2.VideoCapture(video_path)

        stframe = st.empty()
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > 300:  # 最多處理 300 幀避免太久
                break
            if frame_count % 10 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = extract_face(rgb_frame)
                if face is not None:
                    face = cv2.resize(face, (256, 256))
                    resnet_label, resnet_conf, custom_label, custom_conf = predict_with_models(face)
                    cv2.putText(frame, f"ResNet50: {resnet_label} ({resnet_conf:.2%})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Custom CNN: {custom_label} ({custom_conf:.2%})", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            frame_count += 1

        cap.release()
        st.success("影片分析結束 ✅")
