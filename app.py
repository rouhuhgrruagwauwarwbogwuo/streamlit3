import streamlit as st
import numpy as np
import cv2
import requests
import os
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# 載入模型的函數，含下載與錯誤處理
@st.cache_resource
def load_models():
    # Custom CNN 模型（從 Hugging Face 下載）
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    model_path = "/tmp/deepfake_cnn_model.h5"

    if not os.path.exists(model_path):
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
        else:
            raise Exception(f"模型下載失敗，HTTP 狀態碼：{response.status_code}")

    if os.path.exists(model_path):
        try:
            custom_model = load_model(model_path)
        except OSError as e:
            raise Exception(f"載入自訂模型失敗：{e}")
    else:
        raise FileNotFoundError(f"找不到模型檔案：{model_path}")

    # ResNet50 模型
    resnet_model = ResNet50(weights="imagenet", include_top=False, pooling='avg')

    return custom_model, resnet_model

# 圖片預處理函數（for Custom CNN）
def preprocess_custom(image: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.merge([enhanced, enhanced, enhanced])
    resized = cv2.resize(enhanced, target_size)
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

# 圖片預處理函數（for ResNet50）
def preprocess_resnet(image: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    resized = cv2.resize(image, target_size)
    array = img_to_array(resized)
    array = np.expand_dims(array, axis=0)
    return preprocess_input(array)

# 預測函數
def predict(image: np.ndarray, custom_model, resnet_model):
    input_custom = preprocess_custom(image)
    input_resnet = preprocess_resnet(image)

    # Custom CNN 預測
    pred_custom = custom_model.predict(input_custom)[0][0]

    # ResNet50 特徵抽取 + 假設自訂分類器判斷（此處略作處理）
    features = resnet_model.predict(input_resnet)
    # 模擬分類分數（僅作展示）
    pred_resnet = float(np.mean(features)) % 1.0

    return pred_custom, pred_resnet

# Streamlit 主程式
def main():
    st.set_page_config(page_title="Deepfake 圖像偵測", layout="centered")
    st.title("🕵️‍♂️ Deepfake 偵測器")

    uploaded_file = st.file_uploader("請上傳圖片", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        st.image(image, caption="上傳的圖片", use_container_width=True)

        with st.spinner("正在載入模型並進行預測..."):
            try:
                custom_model, resnet_model = load_models()
                pred_custom, pred_resnet = predict(image_np, custom_model, resnet_model)

                st.subheader("🔍 預測結果")
                st.write(f"🧠 Custom CNN 預測值：`{pred_custom:.4f}`")
                st.write(f"📷 ResNet50 特徵預測模擬值：`{pred_resnet:.4f}`")

                st.success("✅ 偵測完成")
            except Exception as e:
                st.error(f"❌ 發生錯誤：{str(e)}")

if __name__ == "__main__":
    main()
