import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from PIL import Image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import tensorflow as tf

# 加載模型
resnet_model = ResNet50(weights="imagenet")

# ========= 預處理相關函數 =========

def center_crop(img, target_size):
    width, height = img.size
    new_width, new_height = target_size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return img.crop((left, top, right, bottom))

def apply_fft_highpass(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    fshift[crow - 10:crow + 10, ccol - 10:ccol + 10] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return cv2.cvtColor(img_back, cv2.COLOR_GRAY2RGB)

def unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.abs(img - blurred) < threshold
        np.copyto(sharpened, img, where=low_contrast_mask)
    return sharpened

def preprocess_for_resnet(img):
    img = img.resize((800, 800), Image.Resampling.LANCZOS)
    img = center_crop(img, (224, 224))
    img_array = np.array(img)

    img_array = apply_fft_highpass(img_array)
    img_array = unsharp_mask(img_array)
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0)

    resnet_input = preprocess_input(np.expand_dims(img_array, axis=0))
    return resnet_input

# ========== 預測函數 ===========
def predict_with_resnet(img):
    img_array = preprocess_for_resnet(img)
    predictions = resnet_model.predict(img_array)
    top_pred = decode_predictions(predictions, top=1)[0][0]
    label = top_pred[1]
    confidence = float(top_pred[2])
    if confidence > 0.9:
        result_label = "real"
    else:
        result_label = "deepfake"
    return result_label, confidence

# ========== Streamlit App ===========
st.set_page_config(page_title="Deepfake 偵測系統", layout="centered")
st.title("🔍 Deepfake 偵測系統（使用 ResNet50）")

uploaded_file = st.file_uploader("請上傳圖片檔案", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert('RGB')
    st.image(pil_img, caption="上傳的圖片", use_container_width=True)

    label, confidence = predict_with_resnet(pil_img)

    st.subheader("📌 偵測結果：")
    st.write(f"判斷為：**{label.upper()}**")
    st.write(f"信心分數：{confidence:.4f}")
