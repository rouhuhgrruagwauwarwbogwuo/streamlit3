import streamlit as st
import numpy as np
import cv2
import tempfile
import requests
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# ==== 設置頁面配置 ====
st.set_page_config(page_title="Deepfake 偵測器", layout="wide")

# 🔹 下載模型（只會下載一次）
@st.cache_resource
def download_model():
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    model_path = "deepfake_cnn_model.h5"
    if not os.path.exists(model_path):
        with st.spinner("下載模型中..."):
            r = requests.get(model_url)
            with open(model_path, "wb") as f:
                f.write(r.content)
    return load_model(model_path)

# 🔹 載入 ResNet50 模型
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
resnet_classifier = Sequential([
    resnet_model,
    Dense(1, activation='sigmoid')  # 1 個輸出節點（0: 真實, 1: 假）
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔹 載入自訂 CNN 模型
custom_model = download_model()

# 🔹 去噪 + 光線標準化的預處理函數
def preprocess_image(image_path, target_size=(256, 256)):
    try:
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img).astype('uint8')

        # 轉換成灰階
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)
        
        # 轉回 3 通道
        img_array = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        # 標準化影像 (0~1)
        img_array = img_array / 255.0
        
        return np.expand_dims(img_array, axis=0)  # 增加 batch 維度
    
    except Exception as e:
        print(f"發生錯誤：{e}")
        return None

# 🔹 預處理圖片，確保 ResNet 和 自訂 CNN 都能處理
def preprocess_for_both_models(image_path):
    img = image.load_img(image_path, target_size=(256, 256))  # 調整大小
    img_array = image.img_to_array(img)
    
    # ResNet50 需要特別的 preprocess_input
    resnet_input = preprocess_input(np.expand_dims(img_array, axis=0))
    
    # 自訂 CNN 只需要正規化 (0~1)
    custom_input = np.expand_dims(img_array / 255.0, axis=0)
    
    return resnet_input, custom_input

# 🔹 進行預測
def predict_with_both_models(image_path):
    try:
        resnet_input, custom_input = preprocess_for_both_models(image_path)
        
        # 檢查預處理後的輸入形狀
        print(f"ResNet input shape: {resnet_input.shape}")
        
        # ResNet50 預測
        resnet_prediction = resnet_classifier.predict(resnet_input)
        if resnet_prediction.ndim > 1:
            resnet_prediction = resnet_prediction[0][0]  # 若返回多維，取出所需的部分
        resnet_label = "Deepfake" if resnet_prediction > 0.5 else "Real"
        
        # 自訂 CNN 模型預測
        custom_prediction = custom_model.predict(custom_input)[0][0]
        custom_label = "Deepfake" if custom_prediction > 0.5 else "Real"
        
        return resnet_label, resnet_prediction, custom_label, custom_prediction
    
    except Exception as e:
        print(f"發生錯誤：{e}")
        return None, None, None, None

# 🔹 顯示圖片和預測結果
def show_prediction(image_path):
    resnet_label, resnet_confidence, custom_label, custom_confidence = predict_with_both_models(image_path)
    
    # 顯示圖片
    img = image.load_img(image_path, target_size=(256, 256))
    st.image(img, caption="原始圖片", use_container_width=True)
    
    # 顯示預測結果
    if resnet_label and custom_label:
        st.subheader(f"ResNet50: {resnet_label} ({resnet_confidence:.2%} 信心分數)")
        st.subheader(f"Custom CNN: {custom_label} ({custom_confidence:.2%} 信心分數)")

# ==== Streamlit App 主體 ====
st.title("🧠 Deepfake 圖片偵測器")

tab1, tab2 = st.tabs(["🖼️ 圖片偵測", "🎥 影片偵測"])

# ---------- 圖片 ----------
with tab1:
    st.header("圖片偵測")
    uploaded_image = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image_path = uploaded_image.name
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        show_prediction(image_path)

# ---------- 影片 ----------
with tab2:
    st.header("影片偵測（每 10 幀抽圖）")
    uploaded_video = st.file_uploader("上傳影片", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        st.info("🎬 擷取影片幀與進行預測中...")

        frames = extract_frames(video_path, interval=10)
        results = predict_frames(resnet_classifier, frames)

        for idx, frame, label, confidence in results:
            st.image(frame, caption=f"第 {idx} 幀 - {label} ({confidence:.2%})", use_container_width=True)
