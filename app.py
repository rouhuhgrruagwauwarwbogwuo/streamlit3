import os
import numpy as np
import cv2
import tempfile
import requests
import h5py
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing import image
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

# 圖像增強（CLAHE + 銳化 + 去噪）
def enhance_image(img):
    # 1. 先進行 CLAHE
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # 2. 銳化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharp = cv2.filter2D(img_eq, -1, kernel)

    # 3. 去噪
    img_denoised = cv2.fastNlMeansDenoisingColored(img_sharp, None, 10, 10, 7, 21)

    return img_denoised

def preprocess_for_models(img):
    img = enhance_image(img)  # 提升畫質
    img_resized = cv2.resize(img, (256, 256))

    # 保留原圖顏色不變的預處理
    efficientnet_input = preprocess_input(np.expand_dims(img_resized, axis=0))
    
    # 使用 CLAHE 增強灰度圖像並還原顏色
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    clahe_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    custom_input = np.expand_dims(clahe_rgb / 255.0, axis=0)
    
    return efficientnet_input, custom_input, img_resized

def smooth_predictions(pred_list, window_size=5):
    if len(pred_list) < window_size:
        return pred_list
    return np.convolve(pred_list, np.ones(window_size)/window_size, mode='valid')

def plot_confidence(eff_conf, custom_conf, combined_conf):
    fig, ax = plt.subplots()
    models = ['EfficientNetB0', 'Custom CNN', 'Combined']
    confs = [eff_conf, custom_conf, combined_conf]
    ax.bar(models, confs, color=['blue', 'green', 'purple'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Confidence')
    st.pyplot(fig)

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
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        efficientnet_input, custom_input, display_img = preprocess_for_models(img)
        eff_pred = efficientnet_classifier.predict(efficientnet_input)[0][0]
        custom_pred = custom_model.predict(custom_input)[0][0]
        combined_pred = (eff_pred + custom_pred) / 2
        label = "Deepfake" if combined_pred > 0.5 else "Real"
        confidence = combined_pred if combined_pred > 0.5 else 1 - combined_pred

        boxed_img = draw_face_box(display_img, label, confidence)
        st.image(boxed_img, caption=f"預測結果：{label} ({confidence:.2%})", use_container_width=True)
        plot_confidence(eff_pred, custom_pred, combined_pred)
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

        frame_preds = []
        frame_count = 0
        while cap.isOpened():
            if st.session_state.get('stop_processing', False):
                st.warning("影片處理已被終止。")
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 == 0:
                try:
                    efficientnet_input, custom_input, display_img = preprocess_for_models(frame)
                    eff_pred = efficientnet_classifier.predict(efficientnet_input)[0][0]
                    custom_pred = custom_model.predict(custom_input)[0][0]
                    combined_pred = (eff_pred + custom_pred) / 2
                    label = "Deepfake" if combined_pred > 0.5 else "Real"
                    confidence = combined_pred if combined_pred > 0.5 else 1 - combined_pred

                    boxed_img = draw_face_box(display_img, label, confidence)
                    st.image(boxed_img, caption=f"幀 {frame_count}: {label} ({confidence:.2%})", use_container_width=True)

                    frame_preds.append(combined_pred)
                except Exception as e:
                    st.error(f"處理幀錯誤: {e}")
                    break

        cap.release()

        if frame_preds:
            smoothed = smooth_predictions(frame_preds)
            st.line_chart(smoothed)
        else:
            st.warning("❌ 沒有有效的幀預測結果。")

        st.success("🎉 偵測完成！")
    except Exception as e:
        st.error(f"❌ 影片處理錯誤: {e}")
        return None

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
