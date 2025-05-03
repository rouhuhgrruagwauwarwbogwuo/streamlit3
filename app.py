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

# 載入 OpenCV 人臉檢測
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 銳化濾波器
def sharpen_image(image):
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

# CLAHE處理
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image

# 增加對比度
def increase_contrast(image, alpha=1.5, beta=50):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# 去噪處理
def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# 提高解析度
def resize_image(image, target_size=(256, 256)):
    return cv2.resize(image, target_size)

# 綜合圖像增強
def enhance_image(image):
    # 提高解析度
    image_resized = resize_image(image)
    
    # CLAHE 增強
    image_clahe = apply_clahe(image_resized)
    
    # 銳化處理
    image_sharpened = sharpen_image(image_clahe)
    
    # 增加對比度
    image_contrast = increase_contrast(image_sharpened)
    
    # 去噪處理
    image_denoised = denoise_image(image_contrast)
    
    return image_denoised

# 圖像預處理：使用人臉 + 增強處理
def preprocess_image(img):
    # 增強圖像清晰度
    enhanced_img = enhance_image(img)
    
    # 人臉檢測
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        face_img = enhanced_img
    else:
        x, y, w, h = faces[0]
        face_img = enhanced_img[y:y+h, x:x+w]

    face_img = cv2.resize(face_img, (256, 256))

    resnet_input = preprocess_input(np.expand_dims(face_img, axis=0).astype(np.float32))
    custom_input = np.expand_dims(face_img / 255.0, axis=0)
    return enhanced_img, resnet_input, custom_input

# 圖片偵測
def process_image(file_bytes):
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

    # 顯示圖片並呈現 ResNet50 和 Custom CNN 的預測
    rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    st.image(rgb_img, caption=f"ResNet50 預測：{resnet_label} ({resnet_confidence:.2%}), Custom CNN 預測：{custom_label} ({custom_confidence:.2%})", use_container_width=True)
    
    return resnet_label, resnet_confidence, custom_label, custom_confidence

# 影片處理（每 10 幀）
def process_video(video_file):
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
            try:
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
            except Exception as e:
                st.warning(f"處理幀錯誤：{e}")
                continue
    cap.release()

    # 顯示信心圖表（ResNet50 和 Custom CNN 的預測結果）
    if resnet_preds:
        st.line_chart(resnet_preds)
    if custom_preds:
        st.line_chart(custom_preds)

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
