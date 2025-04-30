import os
import numpy as np
import cv2
import tempfile
import h5py
import requests
import streamlit as st
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense

# 安裝 OpenCV 頭部版本的安全性處理
try:
    import cv2
except ImportError:
    st.error("❌ 未安裝 OpenCV，正在嘗試安裝 opencv-python-headless...")
    os.system('pip install opencv-python-headless==4.5.5.64')

# ✅ 使用 Hugging Face 的 hf_hub_download 函數
@st.cache_resource
def download_model():
    try:
        model_path = hf_hub_download(
            repo_id="wuwuwu123123/deepfakemodel2",  # 替換成你的 repo ID
            filename="deepfake_cnn_model.h5"         # 模型檔名
        )
        with h5py.File(model_path, 'r') as f:
            pass
        return load_model(model_path)
    except Exception as e:
        st.error("❌ 模型下載或載入失敗，請確認 repo_id 與 filename 正確無誤。")
        raise e

# 載入模型
try:
    custom_model = download_model()
except Exception as e:
    st.error(f"❌ 模型載入失敗: {e}")
    st.stop()

resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
resnet_classifier = Sequential([
    resnet_model,
    Dense(1, activation='sigmoid')
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 圖像增強處理
def enhance_image(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img_sharp = cv2.filter2D(img_eq, -1, kernel)
    return img_sharp

def preprocess_for_models(img):
    img = enhance_image(img)
    img_resized = cv2.resize(img, (256, 256))
    resnet_input = preprocess_input(np.expand_dims(img_resized, axis=0))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    clahe_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    custom_input = np.expand_dims(clahe_rgb / 255.0, axis=0)
    return resnet_input, custom_input, img_resized

def smooth_predictions(pred_list, window_size=5):
    if len(pred_list) < window_size:
        return pred_list
    return np.convolve(pred_list, np.ones(window_size)/window_size, mode='valid')

def plot_confidence(resnet_conf, custom_conf, combined_conf):
    fig, ax = plt.subplots()
    models = ['ResNet50', 'Custom CNN', 'Combined']
    confs = [resnet_conf, custom_conf, combined_conf]
    ax.bar(models, confs, color=['blue', 'green', 'purple'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Confidence')
    st.pyplot(fig)

def process_image(file_bytes):
    try:
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        resnet_input, custom_input, display_img = preprocess_for_models(img)
        resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
        custom_pred = custom_model.predict(custom_input)[0][0]
        combined_pred = (resnet_pred + custom_pred) / 2
        label = "Deepfake" if combined_pred > 0.5 else "Real"
        confidence = combined_pred if combined_pred > 0.5 else 1 - combined_pred
        st.image(img, caption=f"預測結果：{label} ({confidence:.2%})", use_container_width=True)
        plot_confidence(resnet_pred, custom_pred, combined_pred)
    except Exception as e:
        st.error(f"❌ 圖片處理錯誤: {e}")

def process_video_and_generate_result(video_file):
    try:
        temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            st.error("❌ 無法打開影片檔案。")
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.write(f"影片總幀數: {total_frames}")
        
        frame_preds = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 == 0:
                try:
                    resnet_input, custom_input, display_img = preprocess_for_models(frame)
                    resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
                    custom_pred = custom_model.predict(custom_input)[0][0]
                    combined_pred = (resnet_pred + custom_pred) / 2
                    label = "Deepfake" if combined_pred > 0.5 else "Real"
                    confidence = combined_pred if combined_pred > 0.5 else 1 - combined_pred

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(display_img, f"{label} ({confidence:.2%})", (10, 30),
                                font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    st.image(display_img, caption=f"幀 {frame_count}: {label} ({confidence:.2%})", use_container_width=True)

                    frame_preds.append(combined_pred)

                except Exception as e:
                    st.error(f"處理幀錯誤: {e}")
                    break

        cap.release()
        smoothed = smooth_predictions(frame_preds)
        st.line_chart(smoothed)
        st.success("🎉 偵測完成！")
    except Exception as e:
        st.error(f"❌ 影片處理錯誤: {e}")
        return None

# Streamlit UI
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
                st.error("❌ 無法處理影片。")
        else:
            st.warning("請確認上傳的檔案類型與選擇一致。")
    except Exception as e:
        st.error(f"❌ 發生錯誤: {e}")
