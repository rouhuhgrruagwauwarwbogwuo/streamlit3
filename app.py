import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
import requests
from io import BytesIO

st.set_page_config(page_title="Deepfake 偵測", layout="centered")
st.title("Deepfake 偵測工具 (價值增強 + FFT + YCbCr)")

# 載入 ResNet50
resnet_model = ResNet50(weights='imagenet')

# 載入 custom CNN 模型
def load_custom_model_from_huggingface(model_url):
    response = requests.get(model_url)
    model = load_model(BytesIO(response.content))
    return model

# 使用您提供的 Hugging Face 模型 URL
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

# ResNet50 預測
def predict_with_resnet(img_tensor):
    predictions = resnet_model.predict(img_tensor)
    decoded = decode_predictions(predictions, top=3)[0]
    label = decoded[0][1]
    confidence = float(decoded[0][2])
    return label, confidence, decoded

# Custom model 預測
def predict_with_custom_model(img_tensor):
    # 檢查自訓練模型的輸入形狀
    print(f"Custom Model Input Shape: {custom_model.input_shape}")
    
    # 檢查 img_tensor 的形狀
    print(f"Input tensor shape before expand_dims: {img_tensor.shape}")
    
    # 如果模型期望的形狀是 (batch_size, 224, 224, 3)，確保圖片形狀正確
    img_tensor = np.expand_dims(img_tensor, axis=0)  # 添加批次維度
    print(f"Input tensor shape after expand_dims: {img_tensor.shape}")
    
    img_tensor = img_tensor.astype('float32') / 255.0  # 轉換為 [0, 1] 範圍
    print(f"Input tensor dtype: {img_tensor.dtype}")
    
    predictions = custom_model.predict(img_tensor)
    return predictions

# 使用 Streamlit 進行圖片上傳
uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="原始圖片", use_container_width=True)

    # 預處理圖片
    resnet_input, processed_img, cbcr_img = preprocess_advanced(pil_img)

    # 使用 ResNet50 預測
    label, confidence, decoded = predict_with_resnet(resnet_input)

    # 使用 Custom CNN 模型輔助預測
    custom_predictions = predict_with_custom_model(resnet_input)
    custom_label = "deepfake" if custom_predictions[0][0] > 0.5 else "real"
    custom_confidence = custom_predictions[0][0] if custom_predictions[0][0] > 0.5 else 1 - custom_predictions[0][0]

    st.subheader("ResNet50 預測結果")
    st.markdown(f"**Top-1 類別**: `{label}`\n\n**信心度**: `{confidence:.4f}`")

    st.subheader("Custom CNN 預測結果")
    st.markdown(f"**預測類別**: `{custom_label}`\n\n**信心度**: `{custom_confidence:.4f}`")

    st.subheader("預處圖")
    st.image(processed_img, caption="FFT + Unsharp Mask", use_container_width=True)

    st.subheader("CbCr 分頹")
    st.image(cbcr_img, caption="YCbCr - Cb/Cr Channels", use_container_width=True)

    st.markdown("---")
    st.markdown("**Top-3 預測結果 (ResNet50):**")
    for _, name, score in decoded:
        st.write(f"- {name}: {score:.4f}")
