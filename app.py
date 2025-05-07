import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# 載入 ResNet50 預訓練模型
resnet_model = ResNet50(weights="imagenet")

# 使用 MTCNN 進行人臉檢測
detector = MTCNN()

# 提取人臉的函數
def extract_face(pil_img):
    # 將 PIL 圖片轉換為 numpy 陣列
    img_array = np.array(pil_img)

    # 檢測人臉，這將返回人臉的邊界框
    faces = detector.detect_faces(img_array)

    if len(faces) == 0:
        return None  # 若未檢測到人臉，返回 None

    # 假設只有一張人臉，提取該人臉的邊界框
    x, y, width, height = faces[0]['box']

    # 裁切出人臉部分
    face_img = img_array[y:y + height, x:x + width]

    # 將人臉圖片轉換為 PIL 格式
    return Image.fromarray(face_img)

# CLAHE 處理（對比度受限自適應直方圖均衡化）
def apply_clahe(image):
    # 轉換為灰度圖
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # 設定 CLAHE 的參數
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(gray)
    
    # 返回處理過的圖片
    return Image.fromarray(cl_img)

# 銳化處理
def sharpen_image(image):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(2.0)  # 增加銳度，數值越大越銳利

# 預測函數
def predict_with_resnet(img):
    # 1️⃣ 應用圖片處理（CLAHE 和銳化）
    img = apply_clahe(img)  # 應用 CLAHE
    img = sharpen_image(img)  # 應用銳化處理
    
    # 2️⃣ 將圖片縮放至 ResNet50 預期大小 224x224
    img = img.resize((224, 224))

    # 3️⃣ 將圖片轉為 NumPy 陣列並進行 ResNet50 預處理
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 扩展至批次大小 (1, 224, 224, 3)
    img_array = preprocess_input(img_array)  # ResNet50 預處理

    # 4️⃣ 執行預測
    predictions = resnet_model.predict(img_array)

    # 5️⃣ 根據預測結果解釋標籤 (可自訂判定邏輯)
    label = "Deepfake" if predictions[0][0] > 0.5 else "Real"
    confidence = predictions[0][0]

    return label, confidence

# Streamlit 顯示
st.title("Deepfake 偵測 (ResNet50)")

uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)
    st.image(pil_img, caption="上傳的圖片", use_column_width=True)

    # 提取人臉
    face_img = extract_face(pil_img)
    
    if face_img is not None:
        # 顯示提取的人臉圖片
        st.image(face_img, caption="提取的人臉", use_column_width=True)

        # 呼叫預測
        label, confidence = predict_with_resnet(face_img)

        # 顯示結果
        st.write(f"預測結果: {label}")
        st.write(f"信心分數: {confidence:.2f}")
    else:
        st.write("未檢測到人臉")
