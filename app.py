import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# 設置頁面標題與配置
st.set_page_config(page_title="Deepfake 偵測", layout="centered")
st.title("Deepfake 偵測工具")

# 載入 ResNet50 模型
resnet_model = ResNet50(weights='imagenet')

# 圖像中央裁剪
def center_crop(img, target_size=(224, 224)):
    width, height = img.size
    new_width, new_height = target_size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return img.crop((left, top, right, bottom))

# 頻率域高通濾波
def apply_fft_high_pass(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows // 2 , cols // 2
    fshift[crow-20:crow+20, ccol-20:ccol+20] = 0  # 去除低頻部分
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_back

# Unsharp Mask (銳化)
def apply_unsharp_mask(img):
    gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
    unsharp = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    return unsharp

# 顏色空間提取（YCbCr）
def extract_ycbcr_channels(img_array):
    ycbcr = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
    _, cb, cr = cv2.split(ycbcr)
    return cb, cr

# 隨機數據增強（旋轉、平移、裁剪）
def augment_image(img_array):
    # 隨機旋轉
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((img_array.shape[1]//2, img_array.shape[0]//2), angle, 1)
    img_array = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]))
    
    # 隨機平移
    tx = np.random.uniform(-10, 10)
    ty = np.random.uniform(-10, 10)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img_array = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]))
    
    # 隨機裁剪
    h, w, _ = img_array.shape
    top = np.random.randint(0, h // 4)
    left = np.random.randint(0, w // 4)
    bottom = h - np.random.randint(0, h // 4)
    right = w - np.random.randint(0, w // 4)
    img_array = img_array[top:bottom, left:right]
    img_array = cv2.resize(img_array, (224, 224))  # 重設回原大小
    
    return img_array

# 合併預處理（FFT + USM + YCbCr + 數據增強）
def preprocess_advanced(img):
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img = center_crop(img, (224, 224))
    img_array = np.array(img)

    # FFT 高通濾波
    high_pass_img = apply_fft_high_pass(img_array)
    high_pass_img_color = cv2.merge([high_pass_img]*3)  # 換 RGB

    # Unsharp Mask
    enhanced_img = apply_unsharp_mask(high_pass_img_color)

    # 顏色空間提取
    cb, cr = extract_ycbcr_channels(img_array)
    cb = cv2.resize(cb, (224, 224))
    cr = cv2.resize(cr, (224, 224))
    cbcr_3ch = cv2.merge([cb, cr, np.zeros_like(cb)])

    # 增強圖像
    augmented_img = augment_image(enhanced_img)

    # 最終圖像處理
    final_input = preprocess_input(np.expand_dims(augmented_img, axis=0))

    return final_input

# 預測與顯示結果
def predict_image(img):
    processed_img = preprocess_advanced(img)
    prediction = resnet_model.predict(processed_img)
    decoded_pred = decode_predictions(prediction)
    return decoded_pred

# Streamlit 介面
st.subheader("上傳一張圖片進行深偽檢測")

uploaded_file = st.file_uploader("選擇一張圖片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="上傳的圖片", use_container_width=True)
    
    # 預測結果
    pred_result = predict_image(image)

    # 預測的信心分數
    confidence = pred_result[0][0][2]

    # 判斷預測結果
    if confidence > 0.5:
        final_label = "deepfake"
    else:
        final_label = "real"

    # 顯示結果
    st.write(f"預測結果：{final_label}")
    st.write(f"信心分數：{confidence:.2f}")
