import streamlit as st
import numpy as np
import cv2
import requests
from io import BytesIO
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------
# 模型下載與載入
# --------------------------
@st.cache_resource
def load_models():
    # 下載 custom CNN 模型
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    response = requests.get(model_url)
    model_path = "/tmp/deepfake_cnn_model.h5"
    with open(model_path, "wb") as f:
        f.write(response.content)
    custom_model = load_model(model_path)

    # ResNet50 模型 + custom 預測頭
    base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(256, 256, 3))
    resnet_model = Sequential([
        base_model,
        Dense(1, activation='sigmoid')
    ])
    resnet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return custom_model, resnet_model

custom_model, resnet_model = load_models()

# --------------------------
# 預處理
# --------------------------
def preprocess_for_models(pil_img):
    img = pil_img.resize((256, 256))
    img_array = np.array(img).astype(np.uint8)

    # -------- 自訂 CNN 預處理 --------
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    custom_input = np.expand_dims(enhanced_rgb / 255.0, axis=0)

    # -------- ResNet 預處理 --------
    resnet_input = preprocess_input(np.expand_dims(img_array.astype(np.float32), axis=0))

    return custom_input, resnet_input, img_array

# --------------------------
# 額外圖像處理功能（可選）
# --------------------------
def apply_highpass_filter(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F)

def apply_fft(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    spectrum = np.log(np.abs(fshift) + 1)
    return spectrum

def convert_to_ycbcr(img_array):
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)

# --------------------------
# 預測
# --------------------------
def predict(image_pil):
    custom_input, resnet_input, img_array = preprocess_for_models(image_pil)

    resnet_pred = resnet_model.predict(resnet_input)[0][0]
    custom_pred = custom_model.predict(custom_input)[0][0]

    resnet_label = "Deepfake" if resnet_pred > 0.5 else "Real"
    custom_label = "Deepfake" if custom_pred > 0.5 else "Real"

    return resnet_label, resnet_pred, custom_label, custom_pred, img_array

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Deepfake 偵測", layout="wide")
st.title("🕵️ Deepfake 圖片偵測器")
st.markdown("上傳圖片，我們會使用 ResNet50 與自訂 CNN 模型進行判斷")

uploaded_file = st.file_uploader("請選擇一張圖片", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="上傳圖片", use_container_width=True)

    with st.spinner("正在進行分析..."):
        resnet_label, resnet_score, custom_label, custom_score, img_array = predict(image_pil)

    st.success("分析完成 ✅")

    # 顯示預測結果
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔍 ResNet50 預測", resnet_label, f"{resnet_score:.2%}")
    with col2:
        st.metric("🧠 Custom CNN 預測", custom_label, f"{custom_score:.2%}")

    # 顯示信心分數圖表
    st.subheader("📊 模型信心分數")
    fig, ax = plt.subplots(figsize=(6, 3))
    models = ['ResNet50', 'Custom CNN']
    scores = [resnet_score, custom_score]
    ax.bar(models, scores, color=['#1f77b4', '#ff7f0e'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence Scores (Deepfake > 0.5)")
    st.pyplot(fig)

    # 額外圖像處理視覺化
    with st.expander("🧪 額外影像分析（進階）"):
        col1, col2 = st.columns(2)
        with col1:
            st.caption("🔹 高通濾波")
            hp = apply_highpass_filter(img_array)
            st.image(hp, use_container_width=True, clamp=True)

            st.caption("🔹 YCbCr")
            ycbcr = convert_to_ycbcr(img_array)
            st.image(ycbcr, use_container_width=True)

        with col2:
            st.caption("🔹 頻域分析（FFT）")
            fft = apply_fft(img_array)
            st.image(fft, use_container_width=True, clamp=True)
