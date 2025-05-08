import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
import cv2
import tempfile
import pywt
from keras.applications import ResNet50, EfficientNetB0, Xception
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.resnet50 import preprocess_input as preprocess_resnet
from keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from keras.applications.xception import preprocess_input as preprocess_xception
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

# 初始化 MTCNN
st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖像偵測器")
detector = MTCNN()

# 載入模型
@st.cache_resource
def load_models():
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    xception_model = Xception(weights='imagenet', include_top=False, pooling='avg', input_shape=(299, 299, 3))

    resnet_classifier = Sequential([resnet_model, Dense(1, activation='sigmoid')])
    efficientnet_classifier = Sequential([efficientnet_model, Dense(1, activation='sigmoid')])
    xception_classifier = Sequential([xception_model, Dense(1, activation='sigmoid')])

    return {
        'ResNet50': resnet_classifier,
        'EfficientNet': efficientnet_classifier,
        'Xception': xception_classifier
    }

# 提取人臉
@st.cache_data(show_spinner=False)
def extract_face(pil_img):
    img_array = np.array(pil_img)
    faces = detector.detect_faces(img_array)
    if faces:
        x, y, w, h = faces[0]['box']
        face = img_array[y:y+h, x:x+w]
        return Image.fromarray(face)
    return None

# 預處理優化：CLAHE + 銳化
def apply_clahe_sharpen(img):
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 銳化
    blurred = cv2.GaussianBlur(img_clahe, (0, 0), 3)
    sharpened = cv2.addWeighted(img_clahe, 1.5, blurred, -0.5, 0)
    return Image.fromarray(sharpened)

# 高頻濾波處理
def high_pass_filter(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rows, cols = img_gray.shape
    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # 設定高頻濾波器
    center = (rows // 2, cols // 2)
    radius = 30
    mask = np.ones((rows, cols, 2), np.uint8)
    cv2.circle(mask, center, radius, (0, 0, 0), -1)
    
    # 應用濾波
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return np.uint8(np.abs(img_back))

# 小波變換（Wavelet Transform）
def wavelet_transform(img):
    img_np = np.array(img)
    coeffs = pywt.dwt2(img_np, 'bior1.3')  # 這是常用的小波變換
    LL, (LH, HL, HH) = coeffs
    # 返回小波變換的四個部分
    return LL, LH, HL, HH

# 預處理圖像
def preprocess_image(img, model_name):
    img = apply_clahe_sharpen(img)  # 預處理優化加入此行

    if model_name == 'Xception':
        img = img.resize((299, 299))
        img_array = np.array(img).astype(np.float32)
        return preprocess_xception(img_array)
    else:
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32)
        if model_name == 'ResNet50':
            return preprocess_resnet(img_array)
        elif model_name == 'EfficientNet':
            return preprocess_efficientnet(img_array)
    return img_array

# 單模型預測
def predict_model(models, img):
    predictions = []
    for name, model in models.items():
        input_data = preprocess_image(img, name)
        prediction = model.predict(np.expand_dims(input_data, axis=0), verbose=0)
        predictions.append(prediction[0][0])
    return predictions

# 集成預測（簡單平均）
def stacking_predict(models, img):
    preds = predict_model(models, img)
    avg = np.mean(preds)
    return "Deepfake" if avg > 0.5 else "Real", avg

# 顯示預測結果
def show_prediction(img, models):
    label, confidence = stacking_predict(models, img)
    st.image(img, caption="輸入圖像", use_container_width=True)
    st.subheader(f"預測結果：**{label}**")
    st.markdown(f"信心分數：**{confidence:.2f}**")

    # 顯示信心分數條
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.barh([0], confidence, color='green' if label == "Real" else 'red')
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('信心分數')
    st.pyplot(fig)

# UI Tab
models = load_models()
tab1, tab2 = st.tabs(["🖼️ 圖像偵測", "🎥 影片偵測"])

with tab1:
    st.header("上傳圖像進行 Deepfake 偵測")
    uploaded_image = st.file_uploader("選擇一張圖像", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="原始圖像", use_container_width=True)

        # 高頻濾波處理
        filtered_img = high_pass_filter(np.array(pil_img))
        st.image(filtered_img, caption="高頻濾波後的圖像", use_container_width=True)

        # 小波變換
        LL, LH, HL, HH = wavelet_transform(pil_img)
        st.image(LL, caption="小波變換 LL 部分", use_container_width=True)
        st.image(HH, caption="小波變換 HH 部分", use_container_width=True)

        face_img = extract_face(pil_img)
        if face_img:
            st.image(face_img, caption="偵測到人臉", width=300)
            show_prediction(face_img, models)
        else:
            st.info("⚠️ 沒偵測到人臉，使用整張圖像預測")
            show_prediction(pil_img, models)

with tab2:
    st.header("影片偵測（處理前幾幀）")
    uploaded_video = st.file_uploader("選擇一段影片", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        st.info("🎬 正在分析影片...（取前 10 幀）")
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        shown = False
        max_frames = 10
        frame_confidences = []

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 3 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb)

                # 高頻濾波處理
                filtered_frame = high_pass_filter(rgb)
                st.image(filtered_frame, caption=f"第 {frame_idx} 幀 高頻濾波後", width=300)

                # 小波變換
                LL, LH, HL, HH = wavelet_transform(pil_frame)
                st.image(LL, caption=f"第 {frame_idx} 幀 小波變換 LL 部分", width=300)

                face_img = extract_face(pil_frame)
                if face_img:
                    st.image(face_img, caption=f"第 {frame_idx} 幀人臉", width=300)
                    label, confidence = stacking_predict(models, face_img)
                    st.subheader(f"預測結果：**{label}**")
                    frame_confidences.append(confidence)
                    shown = True
                    if len(frame_confidences) == 10:
                        avg_confidence = np.mean(frame_confidences)
                        st.markdown(f"影片總體信心分數：**{avg_confidence:.2f}**")
                        break
            frame_idx += 1

        cap.release()
        if not shown:
            st.warning("⚠️ 沒有偵測到人臉。")
