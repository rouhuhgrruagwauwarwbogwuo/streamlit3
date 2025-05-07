import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image as keras_image
from mtcnn import MTCNN

# 初始化模型
resnet_model = ResNet50(weights='imagenet')
detector = MTCNN()

st.title("🔍 Deepfake 偵測（ResNet50）")

# 中心裁切並 resize
def center_crop_and_resize(img, target_size=(224, 224)):
    width, height = img.size
    new_short = min(width, height)
    left = (width - new_short) // 2
    top = (height - new_short) // 2
    right = left + new_short
    bottom = top + new_short
    img_cropped = img.crop((left, top, right, bottom))
    return img_cropped.resize(target_size)

# CLAHE 增強圖像細節
def enhance_image(img):
    img_cv = np.array(img)
    img_yuv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_yuv[:, :, 0])
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(img_eq)

# 偵測並擷取人臉（回傳第一張臉）
def extract_face(pil_img):
    img_np = np.array(pil_img)
    faces = detector.detect_faces(img_np)
    if faces:
        x, y, w, h = faces[0]['box']
        face = img_np[y:y+h, x:x+w]
        return Image.fromarray(face)
    return pil_img  # 若無偵測到人臉則傳回原圖

# ResNet 預測
def predict_with_resnet(img):
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds = resnet_model.predict(img_array)
    decoded = decode_predictions(preds, top=1)[0][0]
    label = decoded[1]
    confidence = float(decoded[2])
    return label, confidence

# UI
uploaded_file = st.file_uploader("📤 上傳圖片", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="原始圖片", use_container_width=True)

    face_img = extract_face(pil_img)
    st.image(face_img, caption="擷取人臉", use_container_width=True)

    # 中心裁切與圖像增強
    face_img = center_crop_and_resize(face_img, (224, 224))
    face_img = enhance_image(face_img)

    # 預測
    label, confidence = predict_with_resnet(face_img)

    st.subheader("🔎 預測結果")
    st.write(f"**類別**：{label}")
    st.write(f"**信心分數**：{confidence:.2f}")
