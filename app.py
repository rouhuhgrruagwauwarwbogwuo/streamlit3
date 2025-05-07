import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mtcnn

# Streamlit 設定
st.set_page_config(page_title="Deepfake 偵測", layout="centered")
st.title("Deepfake 偵測工具 (CLAHE + 資料增強 + MTCNN)")

# 載入 ResNet50 模型
resnet_model = ResNet50(weights='imagenet')

# 使用 MTCNN 來偵測人臉
detector = mtcnn.MTCNN()

# CLAHE（對比度限制自適應直方圖均衡化）
def apply_clahe(img_array):
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)  # 轉換為LAB顏色空間
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)  # 只對L通道進行CLAHE增強
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)  # 轉換回RGB顏色空間

# 進行資料增強
def augment_image(img_array):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # 擴展維度以符合資料增強的需求
    img_array = np.expand_dims(img_array, axis=0)
    # 增強後的圖像
    augmented_image = datagen.random_transform(img_array[0])
    return augmented_image

# 提取人臉區域
def extract_face(img_array):
    faces = detector.detect_faces(img_array)
    if faces:
        x, y, w, h = faces[0]['box']
        face = img_array[y:y+h, x:x+w]
        return face
    return img_array  # 若沒有偵測到人臉則返回原圖

# 預處理圖片（包括 CLAHE, 資料增強和人臉檢測）
def preprocess_image(img):
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img_array = np.array(img)

    # CLAHE增強
    clahe_img = apply_clahe(img_array)

    # 資料增強
    augmented_img = augment_image(clahe_img)

    # 提取人臉區域
    face_img = extract_face(augmented_img)

    # 調整為ResNet50輸入的尺寸
    final_input = preprocess_input(np.expand_dims(face_img, axis=0))
    
    return final_input, augmented_img, face_img

# ResNet50 預測
def predict_with_resnet(img_tensor):
    predictions = resnet_model.predict(img_tensor)
    decoded = decode_predictions(predictions, top=3)[0]
    label = decoded[0][1]
    confidence = float(decoded[0][2])
    return label, confidence, decoded

# Streamlit 上傳圖片
uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="原始圖片", use_container_width=True)

    # 預處理圖片
    resnet_input, augmented_img, face_img = preprocess_image(pil_img)

    # 使用 ResNet50 預測
    label, confidence, decoded = predict_with_resnet(resnet_input)

    # 顯示預測結果
    st.subheader("預測結果")
    st.markdown(f"**Top-1 類別**: `{label}`\n\n**信心度**: `{confidence:.4f}`")

    # 顯示預處理後的圖片
    st.subheader("增強後的圖片")
    st.image(augmented_img, caption="CLAHE + 資料增強", use_container_width=True)

    # 顯示人臉區域
    st.subheader("人臉區域")
    st.image(face_img, caption="提取的人臉區域", use_container_width=True)

    # 顯示 Top-3 預測結果
    st.markdown("---")
    st.markdown("**Top-3 預測結果:**")
    for _, name, score in decoded:
        st.write(f"- {name}: {score:.4f}")
