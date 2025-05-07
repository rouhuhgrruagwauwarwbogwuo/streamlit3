import numpy as np
import streamlit as st
from tensorflow.keras.applications import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from PIL import Image, ImageEnhance, ImageFilter
import os
import requests
from mtcnn import MTCNN

# 🔹 載入 ResNet50 模型
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

# 🔹 CLAHE 預處理
def apply_clahe(image):
    enhancer = ImageEnhance.Contrast(image)
    image_clahe = enhancer.enhance(2)  # 增加對比度
    return image_clahe

# 🔹 頻域分析 (FFT)
def apply_fft(image):
    img_gray = image.convert('L')  # 轉換為灰階圖像
    img_array = np.array(img_gray)

    # 計算傅立葉變換
    f = np.fft.fft2(img_array)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)

    # 轉換回圖像
    magnitude_spectrum_img = Image.fromarray(np.uint8(magnitude_spectrum * 255 / magnitude_spectrum.max()))
    return magnitude_spectrum_img

# 🔹 圖片預處理（包括 CLAHE 和 FFT）
def preprocess_for_resnet(img):
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img = apply_clahe(img)  # CLAHE 處理
    img = apply_fft(img)    # 頻域處理
    img = center_crop(img, (224, 224))  # 中心裁切

    # 轉換為 numpy 陣列，並將數值範圍調整到 [0, 1]
    img_array = np.array(img)
    img_array = img_array.astype(np.float32) / 255.0

    # 擴展維度以符合模型輸入要求： (batch_size, height, width, channels)
    if img_array.ndim == 3:  # 若圖片是 RGB 彩圖 (H, W, C)
        resnet_input = np.expand_dims(img_array, axis=0)
    else:  # 若圖片是灰階圖 (H, W)
        resnet_input = np.expand_dims(img_array, axis=-1)
        resnet_input = np.repeat(resnet_input, 3, axis=-1)  # 重複通道使其符合 RGB

    # 使用 ResNet50 預處理
    resnet_input = preprocess_input(resnet_input)

    # 檢查輸入形狀
    print(f"ResNet 輸入形狀: {resnet_input.shape}")
    
    return resnet_input

# 🔹 ResNet50 預測
def predict_with_resnet(img):
    resnet_input = preprocess_for_resnet(img)
    resnet_prediction = resnet_classifier.predict(resnet_input)[0][0]
    resnet_label = "Deepfake" if resnet_prediction > 0.5 else "Real"
    return resnet_label, resnet_prediction

# 🔹 顯示預測結果
def show_prediction(img):
    resnet_label, resnet_confidence = predict_with_resnet(img)
    st.image(img, caption="原始圖片", use_container_width=True)
    st.subheader(f"ResNet50: {resnet_label} ({resnet_confidence:.2%})")

# 🔹 Streamlit 主頁面
st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖片偵測器")

# ---------- 圖片 ---------- 
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
