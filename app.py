import os
import numpy as np
import cv2
import tempfile
import requests
import h5py
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense
from mtcnn.mtcnn import MTCNN  # ✅ 使用 MTCNN

# Hugging Face 模型下載網址
MODEL_URL = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"

@st.cache_resource
def download_model():
    model_path = os.path.join(tempfile.gettempdir(), "deepfake_cnn_model.h5")
    if not os.path.exists(model_path):
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
        else:
            st.error("❌ 模型下載失敗。")
            raise Exception("模型下載失敗。")
    try:
        with h5py.File(model_path, 'r') as f:
            pass
    except OSError:
        st.error("❌ 模型檔案無法讀取。")
        raise
    return load_model(model_path)

# 載入模型
try:
    custom_model = download_model()
except Exception as e:
    st.error(f"❌ 模型載入失敗: {e}")
    st.stop()

efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
efficientnet_classifier = Sequential([
    efficientnet_model,
    Dense(1, activation='sigmoid')
])
efficientnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

detector = MTCNN()  # ✅ 初始化 MTCNN 偵測器

# ✅ MTCNN 偵測人臉並框住
def draw_face_box(img, label, confidence):
    results = detector.detect_faces(img)
    color = (0, 0, 255) if label == "Deepfake" else (0, 255, 0)
    for result in results:
        x, y, w, h = result['box']
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        text = f"{label} ({confidence:.2%})"
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img

def process_image(file_bytes):
    try:
        # 直接解碼並顯示原始圖片
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 顯示原始圖片
        st.image(img, caption="上傳的原始圖片", use_container_width=True)

        # 進行預測
        img_resized = cv2.resize(img, (256, 256))
        eff_input = preprocess_input(np.expand_dims(img_resized, axis=0))
        custom_input = np.expand_dims(img_resized / 255.0, axis=0)
        
        # 預測
        eff_pred = efficientnet_classifier.predict(eff_input)[0][0]
        custom_pred = custom_model.predict(custom_input)[0][0]
        combined_pred = (eff_pred + custom_pred) / 2
        label = "Deepfake" if combined_pred > 0.5 else "Real"
        confidence = combined_pred if combined_pred > 0.5 else 1 - combined_pred

        # 顯示預測結果
        boxed_img = draw_face_box(img, label, confidence)
        st.image(boxed_img, caption=f"預測結果：{label} ({confidence:.2%})", use_container_width=True)
        
        plot_confidence(eff_pred, custom_pred, combined_pred)
    except Exception as e:
        st.error(f"❌ 圖片處理錯誤: {e}")

# 🔹 Streamlit UI
st.title("🕵️ Deepfake 偵測 App")
option = st.radio("請選擇檔案類型：", ("圖片", "影片"))

uploaded_file = st.file_uploader("📤 上傳檔案", type=["jpg", "jpeg", "png", "mp4", "mov"])

if uploaded_file is not None:
    try:
        if option == "圖片" and uploaded_file.type.startswith("image"):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            process_image(file_bytes)
        elif option == "影片" and uploaded_file.type.startswith("video"):
            st.markdown("### 處理影片中...")
            processed_video_path = process_video_and_generate_result(uploaded_file)
            if processed_video_path:
                st.video(processed_video_path)
        else:
            st.warning("請確認上傳的檔案類型與選擇一致。")
    except Exception as e:
        st.error(f"❌ 發生錯誤: {e}")
