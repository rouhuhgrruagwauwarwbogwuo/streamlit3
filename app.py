import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from mtcnn.mtcnn import MTCNN
import tempfile
import os

# 標題
st.title("Deepfake 偵測系統")
st.write("請上傳圖片或影片進行分析：")

# 載入模型
@st.cache_resource
def load_models():
    resnet_model = load_model("resnet50_deepfake_model.h5", compile=False)
    custom_model = load_model("deepfake_cnn_model.h5", compile=False)
    return resnet_model, custom_model

resnet_classifier, custom_model = load_models()

# 抽取臉部
def extract_face(image):
    detector = MTCNN()
    img_array = np.asarray(image)
    results = detector.detect_faces(img_array)
    if results:
        x, y, w, h = results[0]['box']
        face = img_array[y:y+h, x:x+w]
        return Image.fromarray(face)
    else:
        return image

# 中央裁切
def center_crop(img, size=(224, 224)):
    width, height = img.size
    new_width, new_height = size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return img.crop((left, top, right, bottom))

# 預處理圖片，確保 ResNet 和 自訂 CNN 都能處理
def preprocess_for_both_models(img):
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img = center_crop(img, (224, 224))
    img_array = np.array(img)
    apply_blur = True
    if apply_blur:
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
    resnet_input = preprocess_input(np.expand_dims(img_array, axis=0))
    custom_input = np.expand_dims(img_array / 255.0, axis=0)
    if custom_input.shape[1] != 224 or custom_input.shape[2] != 224:
        custom_input = cv2.resize(custom_input[0], (224, 224))
        custom_input = np.expand_dims(custom_input, axis=0)
    return resnet_input, custom_input

# 預測
def predict_with_both_models(img):
    resnet_input, custom_input = preprocess_for_both_models(img)
    resnet_prediction = resnet_classifier.predict(resnet_input)[0][0]
    resnet_label = "Deepfake" if resnet_prediction > 0.5 else "Real"
    custom_prediction = custom_model.predict(custom_input)[0][0] if custom_model else 0
    custom_label = "Deepfake" if custom_prediction > 0.5 else "Real"
    return resnet_label, resnet_prediction, custom_label, custom_prediction

# 顯示預測結果
def show_prediction(img):
    resnet_label, resnet_confidence, custom_label, custom_confidence = predict_with_both_models(img)
    st.subheader("預測結果")
    st.write(f"ResNet50 模型預測：**{resnet_label}** (信心分數: {resnet_confidence:.2f})")
    st.write(f"自訂 CNN 模型預測：**{custom_label}** (信心分數: {custom_confidence:.2f})")

# 使用者上傳圖片
uploaded_file = st.file_uploader("上傳圖片或影片", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = uploaded_file.read()
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="原始圖片", use_container_width=True)
    face_img = extract_face(pil_img)
    st.image(face_img, caption="擷取人臉後圖片", use_container_width=True)
    show_prediction(face_img)
