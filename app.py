import streamlit as st
import numpy as np
import cv2
import os
import tempfile
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

# ✅ 必須是第一個 Streamlit 指令
st.set_page_config(page_title="Deepfake 偵測", layout="centered")

st.title("🧠 Deepfake 偵測系統 (圖片)")
st.markdown("上傳圖片，並使用訓練好的模型判斷是否為 Deepfake。")

# 載入模型（從 Hugging Face 下載）
@st.cache_resource
def load_deepfake_model():
    url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    model_path = os.path.join(tempfile.gettempdir(), "deepfake_cnn_model.h5")
    if not os.path.exists(model_path):
        with open(model_path, "wb") as f:
            f.write(requests.get(url).content)
    return load_model(model_path)

model = load_deepfake_model()

# 圖片預處理
def preprocess_uploaded_image(uploaded_file):
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img_array = img / 255.0
        return np.expand_dims(img_array, axis=0), img
    except Exception as e:
        st.error(f"❌ 圖片預處理失敗：{e}")
        return None, None

# 預測
def predict_image(img_array):
    prediction = model.predict(img_array)[0][0]
    label = "Deepfake" if prediction > 0.5 else "Real"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

# 上傳圖片
uploaded_file = st.file_uploader("請選擇一張圖片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_array, original_img = preprocess_uploaded_image(uploaded_file)

    if img_array is not None:
        label, confidence = predict_image(img_array)

        st.image(original_img, caption=f"預測結果：{label} ({confidence:.2%} 信心分數)", use_container_width=True)
        st.success(f"✅ 預測：{label}，信心分數：{confidence:.2%}")
