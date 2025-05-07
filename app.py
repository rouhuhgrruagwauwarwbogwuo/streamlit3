import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import tensorflow as tf

# 載入模型（使用預設 ImageNet 預訓練模型）
resnet_model = ResNet50(weights='imagenet')

# 圖片中心裁切函數
def center_crop(img, target_size):
    width, height = img.size
    new_width, new_height = target_size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return img.crop((left, top, right, bottom))

# 📌 圖像預處理（針對 ResNet50）
def preprocess_for_resnet(img):
    # 高清圖縮放
    if img.width > 800 or img.height > 800:
        img = img.resize((800, 800), Image.Resampling.LANCZOS)

    # LANCZOS 縮圖再中心裁切
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img = center_crop(img, (224, 224))
    img_array = np.array(img)

    # 加強模糊處理（避免過清晰誤判）
    img_array = cv2.GaussianBlur(img_array, (5, 5), 1.0)

    # 高通濾波保留邊緣
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_array = cv2.filter2D(img_array, -1, kernel)

    # 預處理給 ResNet50
    resnet_input = preprocess_input(np.expand_dims(img_array, axis=0))
    return resnet_input

# 預測函數
def predict_with_resnet(img):
    img_array = preprocess_for_resnet(img)
    predictions = resnet_model.predict(img_array)
    decoded = decode_predictions(predictions, top=1)[0][0]
    label = decoded[1]
    confidence = float(decoded[2])

    # 自定義標籤：你可以依據模型輸出決定
    # 模擬邏輯：以 'mask', 'fake', 'screen', 'monitor' 等視為 deepfake
    deepfake_keywords = ['mask', 'screen', 'monitor', 'projector', 'fake']
    if any(k in label.lower() for k in deepfake_keywords):
        return "deepfake", confidence
    return "real", confidence

# Streamlit App
st.title("📷 Deepfake 圖片偵測 (ResNet50 版本)")

uploaded_file = st.file_uploader("請上傳圖片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert('RGB')
    st.image(pil_img, caption='上傳圖片', use_container_width=True)

    label, confidence = predict_with_resnet(pil_img)

    st.markdown("---")
    st.subheader("🔍 偵測結果")
    st.write(f"🧠 模型判斷：**{label.upper()}**")
    st.write(f"🔢 信心分數：{confidence:.2f}")
