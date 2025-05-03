import streamlit as st
import os
import requests
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image

MODEL_URL = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
MODEL_PATH = "deepfake_cnn_model.h5"

# 嘗試下載模型（若不存在）
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("正在從 Hugging Face 下載模型..."):
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            else:
                st.error(f"下載模型失敗，HTTP 狀態碼：{response.status_code}")
                return None
    return load_model(MODEL_PATH)

# 簡化預處理（不過度處理）
def preprocess(img: Image.Image, target_size=(256, 256)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # 正規化
    return np.expand_dims(img_array, axis=0)

# 預測函數
def predict(model, img: Image.Image):
    x = preprocess(img)
    pred = model.predict(x)[0][0]
    label = "Deepfake" if pred > 0.5 else "Real"
    confidence = pred if pred > 0.5 else 1 - pred
    return label, confidence

# Streamlit 介面
st.title("🧠 Deepfake 圖片偵測器")
st.write("上傳圖片，我們會使用 CNN 模型進行判斷。")

uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="上傳圖片", use_column_width=True)

    model = download_model()
    if model:
        label, confidence = predict(model, img)
        st.markdown(f"### 預測結果：`{label}`")
        st.markdown(f"### 信心分數：`{confidence:.2%}`")
