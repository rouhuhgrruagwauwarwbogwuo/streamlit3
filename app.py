import os
import numpy as np
import cv2
import tempfile
import requests
import h5py
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense
from mtcnn import MTCNN

# 載入 ResNet50 模型
@st.cache_resource
def load_resnet_model():
    resnet_base = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
    model = Sequential([
        resnet_base,
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

resnet_model = load_resnet_model()

# 載入 Custom CNN 模型作為參考
@st.cache_resource
def load_custom_model():
    model_path = os.path.join(tempfile.gettempdir(), "deepfake_cnn_model.h5")
    if not os.path.exists(model_path):
        model_url = "https://huggingface.co/wuwuwu123123/deepfake/resolve/main/deepfake_cnn_model.h5"
        response = requests.get(model_url)
        with open(model_path, "wb") as f:
            f.write(response.content)
    return load_model(model_path)

custom_model = load_custom_model()

# 載入 MTCNN 人臉檢測器
detector = MTCNN()

# 圖像預處理：使用人臉 + CLAHE + 銳化
def preprocess_image(img):
    faces = detector.detect_faces(img)
    
    if len(faces) == 0:
        face_img = img
    else:
        x, y, w, h = faces[0]['box']
        face_img = img[y:y+h, x:x+w]
    
    face_img = cv2.resize(face_img, (256, 256))

    # 輕度 CLAHE 處理
    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    face_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 輕度銳化
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    sharpened = cv2.filter2D(face_img, -1, kernel)

    resnet_input = preprocess_input(np.expand_dims(sharpened, axis=0).astype(np.float32))
    custom_input = np.expand_dims(sharpened / 255.0, axis=0)
    return sharpened, resnet_input, custom_input

# 圖片偵測
def process_image(file_bytes):
    try:
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        display_img, resnet_input, custom_input = preprocess_image(img)
        
        # ResNet50 預測
        resnet_pred = resnet_model.predict(resnet_input)[0][0]
        resnet_label = "Deepfake" if resnet_pred > 0.5 else "Real"
        resnet_confidence = resnet_pred if resnet_pred > 0.5 else 1 - resnet_pred
        
        # Custom CNN 預測（作為參考）
        custom_pred = custom_model.predict(custom_input)[0][0]
        custom_label = "Deepfake" if custom_pred > 0.5 else "Real"
        custom_confidence = custom_pred if custom_pred > 0.5 else 1 - custom_pred

        # 顯示完整圖片並呈現 ResNet50 的預測
        rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        st.image(rgb_img, caption=f"ResNet50 預測：{resnet_label} ({resnet_confidence:.2%})", use_container_width=True)

        # 顯示 Custom CNN 的預測結果，但使用較小的字型
        st.markdown(f"Custom CNN 預測：{custom_label} ({custom_confidence:.2%})", unsafe_allow_html=True)

        return resnet_label, resnet_confidence, custom_label, custom_confidence
    except Exception as e:
        st.error(f"❌ 發生錯誤：{e}")
        return None

# 影片處理（每 10 幀）
def process_video(video_file):
    try:
        temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(temp_video_path)
        frame_count = 0
        resnet_preds = []
        custom_preds = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 10 == 0:
                display_img, resnet_input, custom_input = preprocess_image(frame)
                
                # ResNet50 預測
                resnet_pred = resnet_model.predict(resnet_input)[0][0]
                resnet_preds.append(resnet_pred)
                
                # Custom CNN 預測
                custom_pred = custom_model.predict(custom_input)[0][0]
                custom_preds.append(custom_pred)

                # 顯示每一幀的結果
                resnet_label = "Deepfake" if resnet_pred > 0.5 else "Real"
                custom_label = "Deepfake" if custom_pred > 0.5 else "Real"
                resnet_confidence = resnet_pred if resnet_pred > 0.5 else 1 - resnet_pred
                custom_confidence = custom_pred if custom_pred > 0.5 else 1 - custom_pred

                rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                st.image(rgb_img, caption=f"第 {frame_count} 幀：ResNet50 預測：{resnet_label} ({resnet_confidence:.2%}), Custom CNN 預測：{custom_label} ({custom_confidence:.2%})", use_container_width=True)

        cap.release()

        # 顯示信心圖表（ResNet50 和 Custom CNN 的預測結果）
        if resnet_preds:
            st.line_chart(resnet_preds)
        if custom_preds:
            st.line_chart(custom_preds)
    except Exception as e:
        st.error(f"❌ 發生錯誤：{e}")

# Streamlit UI
st.title("🎬 Deepfake 偵測 App（ResNet50 和 Custom CNN）")
option = st.radio("選擇檔案類型：", ("圖片", "影片"))
uploaded_file = st.file_uploader("上傳圖片或影片", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    try:
        if option == "圖片" and uploaded_file.type.startswith("image"):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            process_image(file_bytes)
        elif option == "影片" and uploaded_file.type.startswith("video"):
            st.info("影片處理中...")
            process_video(uploaded_file)
        else:
            st.warning("請確認上傳的檔案類型與選擇一致。")
    except Exception as e:
        st.error(f"❌ 發生錯誤：{e}")
