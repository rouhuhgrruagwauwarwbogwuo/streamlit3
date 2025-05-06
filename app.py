import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from PIL import Image
from mtcnn import MTCNN
import tempfile
import os
import requests

# 檢查並下載模型檔案
def download_model():
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    model_filename = "deepfake_cnn_model.h5"
    
    # 如果模型檔案不存在，則下載
    if not os.path.exists(model_filename):
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_filename, "wb") as f:
                f.write(response.content)
            print("模型檔案已成功下載！")
        else:
            print(f"下載失敗，狀態碼：{response.status_code}")
            return None
    return model_filename

# 🔹 載入 ResNet50 模型
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
resnet_classifier = Sequential([
    resnet_model,
    Dense(1, activation='sigmoid')  # 1 個輸出節點（0: 真實, 1: 假）
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔹 載入自訂 CNN 模型
model_path = download_model()
if model_path:
    custom_model = load_model(model_path)
else:
    custom_model = None

# 🔹 初始化 MTCNN 人臉檢測器
detector = MTCNN()

# 🔹 預處理函數 - 高通濾波（Edge Enhancement）
def high_pass_filter(img_array):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filtered_img = cv2.filter2D(img_array, -1, kernel)
    return filtered_img

# 🔹 預處理函數 - 頻域特徵分析 (FFT)
def fft_filter(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

# 🔹 顏色空間轉換
def convert_to_ycbcr(img_array):
    img_ycbcr = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
    return img_ycbcr

def convert_to_lab(img_array):
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    return img_lab

# 🔹 CLAHE + 銳化預處理
def preprocess_image(image_path, target_size=(256, 256)):
    try:
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img).astype('uint8')

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        # 轉回 RGB
        img_array = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        img_array = img_array / 255.0  # 標準化影像 (0~1)
        
        # 高通濾波增強
        img_array = high_pass_filter(img_array)
        
        return np.expand_dims(img_array, axis=0)
    
    except Exception as e:
        print(f"發生錯誤：{e}")
        return None

# 🔹 人臉偵測，擷取人臉區域
def extract_face(img):
    try:
        img_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        
        # 確保圖像尺寸足夠大
        if img_rgb.shape[0] < 20 or img_rgb.shape[1] < 20:
            raise ValueError("圖像尺寸過小，無法進行人臉檢測")
        
        faces = detector.detect_faces(img_rgb)
        if len(faces) > 0:
            x, y, width, height = faces[0]['box']
            face = img_rgb[y:y+height, x:x+width]
            return Image.fromarray(face)
        else:
            print("未偵測到人臉")
            return None
    except Exception as e:
        print(f"人臉偵測錯誤: {e}")
        return None

# 🔹 預處理圖片，確保 ResNet 和 自訂 CNN 都能處理
def preprocess_for_both_models(image_path):
    try:
        img = image.load_img(image_path, target_size=(256, 256))  # 調整大小
        img_array = image.img_to_array(img)
        
        # ResNet50 需要特別的 preprocess_input
        resnet_input = preprocess_input(np.expand_dims(img_array, axis=0))
        
        # 自訂 CNN 只需要正規化 (0~1)
        custom_input = np.expand_dims(img_array / 255.0, axis=0)
        
        return resnet_input, custom_input
    except Exception as e:
        print(f"圖片處理錯誤: {e}")
        return None, None

# 🔹 進行預測
def predict_with_both_models(image_path):
    resnet_input, custom_input = preprocess_for_both_models(image_path)
    
    if resnet_input is None or custom_input is None:
        return "處理錯誤", 0.0, "處理錯誤", 0.0
    
    # ResNet50 預測
    resnet_prediction = resnet_classifier.predict(resnet_input)[0][0]
    resnet_label = "Deepfake" if resnet_prediction > 0.5 else "Real"
    
    # 自訂 CNN 模型預測
    custom_prediction = custom_model.predict(custom_input)[0][0]
    custom_label = "Deepfake" if custom_prediction > 0.5 else "Real"
    
    return resnet_label, resnet_prediction, custom_label, custom_prediction

# 🔹 顯示圖片和預測結果
def show_prediction(image_path):
    resnet_label, resnet_confidence, custom_label, custom_confidence = predict_with_both_models(image_path)
    
    # 顯示圖片
    img = image.load_img(image_path, target_size=(256, 256))
    st.image(img, caption="預測圖片", use_column_width=True)
    
    # 顯示預測結果
    st.subheader(f"ResNet50: {resnet_label} ({resnet_confidence:.2%})\n"
                 f"Custom CNN: {custom_label} ({custom_confidence:.2%})")

# 🔹 Streamlit 主應用程式
st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖片與影片偵測器")

tab1, tab2 = st.tabs(["🖼️ 圖片偵測", "🎥 影片偵測"])

# ---------- 圖片 ---------- 
with tab1:
    st.header("圖片偵測")
    uploaded_image = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="原始圖片", use_column_width=True)

        # 嘗試擷取人臉區域
        face_img = extract_face(pil_img)
        if face_img:
            st.image(face_img, caption="偵測到的人臉", use_column_width=True)
            show_prediction(face_img)
        else:
            st.write("未偵測到人臉，使用整體圖片進行預測")
            show_prediction(uploaded_image)

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
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        results = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 10 == 0:  # 每 10 幀進行一次處理
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                face_img = extract_face(frame_pil)
                if face_img:
                    result = predict_with_both_models(face_img)
                    results.append((frame_idx, result))
                frame_idx += 1
        cap.release()

        # 顯示影片結果
        for idx, (resnet_label, resnet_confidence, custom_label, custom_confidence) in results:
            st.image(frame_pil, caption=f"第 {idx} 幀 - {resnet_label} ({resnet_confidence:.2%})", use_column_width=True)
