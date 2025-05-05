import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import requests
import os
from io import BytesIO
from mtcnn import MTCNN  # 使用 MTCNN 進行人臉偵測

# 🔹 從 Hugging Face 下載模型
model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
response = requests.get(model_url)

# 將模型從 URL 下載並加載
model_path = '/tmp/deepfake_cnn_model.h5'
with open(model_path, 'wb') as f:
    f.write(response.content)

# 載入自訂 CNN 模型
custom_model = load_model(model_path)

# 🔹 載入 ResNet50 模型
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
resnet_classifier = Sequential([
    resnet_model,
    Dense(1, activation='sigmoid')  # 1 個輸出節點（0: 真實, 1: 假）
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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

# 🔹 使用 MTCNN 偵測人臉
def extract_face(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]['box']
        face_img = image[y:y+h, x:x+w]
        return face_img
    else:
        return None

# 🔹 高通濾波
def apply_highpass_filter(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    highpass = cv2.Laplacian(gray_img, cv2.CV_64F)
    return highpass

# 🔹 頻域分析 (FFT)
def apply_fft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

# 🔹 顏色空間轉換 (YCbCr)
def convert_to_ycbcr(image):
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    return ycbcr_image

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
    resnet_input, custom_input = preprocess_for_both_models(image_path)
    
    # ResNet50 預測
    resnet_prediction = resnet_classifier.predict(resnet_input)[0][0]
    resnet_label = "Deepfake" if resnet_prediction > 0.5 else "Real"
    
    # 自訂 CNN 模型預測
    custom_prediction = custom_model.predict(custom_input)[0][0]
    custom_label = "Deepfake" if custom_prediction > 0.5 else "Real"
    
    return resnet_label, resnet_prediction, custom_label, custom_prediction

# 🔹 顯示圖片和預測結果
def show_prediction(image_path):
    # 嘗試擷取人臉
    img = cv2.imread(image_path)
    face_img = extract_face(img)
    
    if face_img is not None:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # 轉為 RGB 格式
        resnet_label, resnet_confidence, custom_label, custom_confidence = predict_with_both_models(face_img)
    else:
        resnet_label, resnet_confidence, custom_label, custom_confidence = predict_with_both_models(image_path)
    
    # 顯示圖片
    img = image.load_img(image_path, target_size=(256, 256))
    st.image(img, caption="上傳的圖片", use_container_width=True)
    
    # 顯示預測結果
    st.write(f"**ResNet50 預測結果**: {resnet_label} ({resnet_confidence:.2%})")
    st.write(f"**自訂 CNN 預測結果**: {custom_label} ({custom_confidence:.2%})")

# 🔹 逐幀處理影片
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 處理每一幀
        face_img = extract_face(frame)
        if face_img is not None:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # 轉為 RGB 格式
            resnet_label, resnet_confidence, custom_label, custom_confidence = predict_with_both_models(face_img)
        else:
            resnet_label, resnet_confidence, custom_label, custom_confidence = predict_with_both_models(frame)
        
        # 顯示預測結果於每一幀
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"ResNet50: {resnet_label} ({resnet_confidence:.2%})", (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Custom CNN: {custom_label} ({custom_confidence:.2%})", (10, 70), font, 1, (0, 255, 0), 2)
        
        # 顯示處理後的幀
        cv2.imshow('Deepfake Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 停止
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 🔹 使用圖片上傳
uploaded_image = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # 儲存上傳的圖片
    image_path = f"temp_image.{uploaded_image.name.split('.')[-1]}"
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    
    # 顯示圖片預測結果
    show_prediction(image_path)

# 🔹 使用影片上傳
uploaded_video = st.file_uploader("上傳影片", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # 儲存上傳的影片
    video_path = f"temp_video.{uploaded_video.name.split('.')[-1]}"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    
    # 逐幀處理影片
    process_video(video_path)
