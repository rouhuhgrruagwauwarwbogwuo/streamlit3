import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import os
from tensorflow.keras.models import load_model
import requests
from io import BytesIO

# 載入 ResNet50 和自訓練的模型
resnet_model = ResNet50(weights='imagenet')

# 從 Hugging Face 下載自訓練模型
def load_custom_model_from_huggingface(model_url):
    response = requests.get(model_url)
    model = load_model(BytesIO(response.content))
    return model

# 假設自訓練模型在 Hugging Face 上
custom_model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
custom_model = load_custom_model_from_huggingface(custom_model_url)

# 圖像中央補白補滿 target size
def center_crop(img, target_size=(224, 224)):
    width, height = img.size
    new_width, new_height = target_size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return img.crop((left, top, right, bottom))

# 對圖片執行 FFT 與高速遮漏
def apply_fft_high_pass(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows // 2 , cols // 2
    fshift[crow-20:crow+20, ccol-20:ccol+20] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_back

# Unsharp Mask 提升詳細
def apply_unsharp_mask(img):
    gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
    unsharp = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    return unsharp

# YCbCr 分析出 Cb 與 Cr
def extract_ycbcr_channels(img_array):
    ycbcr = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
    _, cb, cr = cv2.split(ycbcr)
    return cb, cr

# 合併預處理 (FFT + USM + YCbCr)
def preprocess_advanced(img):
    # 大幅縮放 保護詳細
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img = center_crop(img, (224, 224))
    img_array = np.array(img)

    # FFT 高速遮漏
    high_pass_img = apply_fft_high_pass(img_array)
    high_pass_img_color = cv2.merge([high_pass_img]*3)  # 換 RGB

    # Unsharp Mask
    enhanced_img = apply_unsharp_mask(high_pass_img_color)

    # YCbCr 提取 Cb/Cr
    cb, cr = extract_ycbcr_channels(img_array)
    cb = cv2.resize(cb, (224, 224))
    cr = cv2.resize(cr, (224, 224))
    cbcr_3ch = cv2.merge([cb, cr, np.zeros_like(cb)])

    # 接合所有這些元素 (最終對上 ResNet50)
    final_input = preprocess_input(np.expand_dims(enhanced_img, axis=0))

    return final_input, enhanced_img, cbcr_3ch

# 去除黑色區域的處理（替換黑色區域為白色）
def remove_black_background(img_array):
    # 假設黑色區域為 0
    img_array[img_array == 0] = 255
    return img_array

# ResNet50 預測
def predict_with_resnet(img_tensor):
    predictions = resnet_model.predict(img_tensor)
    decoded = decode_predictions(predictions, top=3)[0]
    label = decoded[0][1]
    confidence = float(decoded[0][2])
    return label, confidence, decoded

# 自訓練模型預測（作為輔助參考）
def predict_with_custom_model(img_tensor):
    predictions = custom_model.predict(img_tensor)
    return predictions

# 上傳圖片
uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="原始圖片", use_container_width=True)

    # 預處理圖片
    resnet_input, processed_img, cbcr_img = preprocess_advanced(pil_img)

    # 去除黑色區域
    processed_img_no_black = remove_black_background(processed_img)

    # 預測 ResNet50
    resnet_label, resnet_confidence, _ = predict_with_resnet(resnet_input)

    # 自訓練模型預測（僅作為輔助參考）
    custom_predictions = predict_with_custom_model(resnet_input)

    # 假設自訓練模型返回的信心分數是 0 到 1 之間
    custom_confidence = custom_predictions[0][0]

    # 顯示最終結果
    st.subheader("最終預測結果 (ResNet50 判斷)")
    st.markdown(f"**分類結果**: `{resnet_label}`\n\n**信心度**: `{resnet_confidence:.4f}`")

    # 顯示自訓練模型的預測結果（作為參考）
    st.subheader("自訓練模型預測結果 (作為參考)")
    st.markdown(f"**自訓練模型信心度**: `{custom_confidence:.4f}`")

    # 顯示去除黑色區域後的預處理圖片
    st.subheader("預處理後的圖片 (黑色區域已去除)")
    st.image(processed_img_no_black, caption="去除黑色區域後的圖片", use_container_width=True)

    # 顯示 CbCr 分析結果
    st.subheader("CbCr 分析")
    st.image(cbcr_img, caption="YCbCr - Cb/Cr Channels", use_container_width=True)

    # 顯示 Top-3 預測結果
    st.markdown("---")
    st.markdown("**Top-3 預測結果 (ResNet50):**")
    for _, name, score in _:
        st.write(f"- {name}: {score:.4f}")
