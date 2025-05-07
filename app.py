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
st.title("Deepfake 偵測工具")

# 載入 ResNet50
resnet_model = ResNet50(weights='imagenet')

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
def preprocess_for_high_res(img):
    # 大幅縮放 保護詳細
    img = img.resize((512, 512), Image.Resampling.LANCZOS)  # 高解析度
    img = center_crop(img, (224, 224))  # 再裁剪至 224x224
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

    # 最終輸入處理
    final_input = preprocess_input(np.expand_dims(enhanced_img, axis=0))

    return final_input, enhanced_img, cbcr_3ch

# ResNet50 預測
def predict_with_resnet(img_tensor):
    predictions = resnet_model.predict(img_tensor)
    decoded = decode_predictions(predictions, top=3)[0]
    label = decoded[0][1]
    confidence = float(decoded[0][2])
    return label, confidence, decoded

# 載入自訂模型
def load_custom_model_from_huggingface(model_url):
    response = requests.get(model_url)
    
    # 檢查下載是否成功
    if response.status_code == 200:
        try:
            # 下載模型後，從 bytes 載入模型
            model = load_model(BytesIO(response.content))
            print("模型成功載入")
            return model
        except Exception as e:
            print(f"載入模型失敗: {e}")
            return None
    else:
        print(f"下載失敗，HTTP 狀態碼: {response.status_code}")
        return None

# 自訂模型預測
def predict_with_custom_model(img_tensor):
    predictions = custom_model.predict(img_tensor)
    confidence = predictions[0][0]  # 假設是二分類模型，返回預測信心度
    return confidence

# 載入自訂模型
custom_model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
custom_model = load_custom_model_from_huggingface(custom_model_url)

if custom_model is None:
    print("自訂模型加載失敗，請檢查模型文件或 URL。")
else:
    print("自訂模型加載成功！")

# 用戶上傳圖片
uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="原始圖片", use_container_width=True)

    # 預處理高清圖片
    resnet_input, processed_img, cbcr_img = preprocess_for_high_res(pil_img)
    
    # 使用 ResNet50 預測
    _, confidence_resnet, _ = predict_with_resnet(resnet_input)

    # 使用自訂模型預測
    custom_confidence = 0
    if custom_model:
        custom_confidence = predict_with_custom_model(resnet_input)

    # 判斷結果 (根據自訂模型的信心度)
    final_label = "real" if custom_confidence < 0.5 else "deepfake"

    # 顯示結果
    st.subheader("最終預測結果")
    st.markdown(f"**預測結果**: `{final_label}`")
    st.markdown(f"ResNet50 信心度: {confidence_resnet * 100:.2f}%")
    st.markdown(f"自訂模型信心度: {custom_confidence * 100:.2f}%")
