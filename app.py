import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import matplotlib.pyplot as plt

# 加載 ResNet50 模型
resnet_model = ResNet50(weights='imagenet')

# 假設這是你的自訂 CNN 模型
custom_model = load_model('custom_cnn_model.h5')

# 🔹 偵測臉部
def detect_faces(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    return faces

# 🔹 高通濾波處理
def apply_highpass_filter(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    highpass = cv2.Laplacian(gray_img, cv2.CV_64F)
    return highpass

# 🔹 頻域分析 (FFT)
def apply_fft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

# 🔹 預處理圖片，準備 ResNet50 和自訂 CNN 模型
def preprocess_for_both_models(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    
    # ResNet50 須使用 preprocess_input
    resnet_input = preprocess_input(np.expand_dims(img_array, axis=0))
    
    # 自訂 CNN 需要標準化
    custom_input = np.expand_dims(img_array / 255.0, axis=0)
    
    return resnet_input, custom_input

# 🔹 時間不一致性檢測 (逐幀處理影片)
def process_video_for_inconsistencies(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = detect_faces(frame)
        
        if prev_frame is not None:
            # 計算與前一幀的差異
            frame_diff = cv2.absdiff(prev_frame, frame)
            gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            cv2.imshow("Frame Difference", thresh)
        
        prev_frame = frame
        
        cv2.imshow('Video Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 🔹 主程序處理圖片
def process_image(image_path):
    # 預處理圖像
    resnet_input, custom_input = preprocess_for_both_models(image_path)

    # 使用 ResNet50 進行預測
    resnet_pred = resnet_model.predict(resnet_input)
    resnet_pred_class = np.argmax(resnet_pred, axis=1)

    # 使用自訂 CNN 模型進行預測
    custom_pred = custom_model.predict(custom_input)
    custom_pred_class = np.argmax(custom_pred, axis=1)

    # 顯示結果
    st.image(image_path, use_column_width=True)
    st.write(f"ResNet50 預測結果: {resnet_pred_class}")
    st.write(f"自訂 CNN 預測結果: {custom_pred_class}")

    # 顯示圖片的高通濾波結果
    image = cv2.imread(image_path)
    highpass_image = apply_highpass_filter(image)
    st.image(highpass_image, caption='高通濾波處理過的圖片', use_column_width=True)

    # 顯示頻域分析圖
    magnitude_spectrum = apply_fft(image)
    st.image(magnitude_spectrum, caption='頻域分析結果', use_column_width=True)

# 🔹 主程序處理影片
def process_video(video_path):
    st.write("影片處理中...")
    process_video_for_inconsistencies(video_path)

# Streamlit UI
st.title("Deepfake 偵測")

st.write("請上傳圖片或影片進行偵測：")

uploaded_file = st.file_uploader("選擇圖片或影片", type=["jpg", "png", "mp4", "mov"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ['jpg', 'png']:
        # 儲存圖片並顯示處理結果
        image_path = f"temp_image.{file_extension}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        process_image(image_path)
    
    elif file_extension in ['mp4', 'mov']:
        # 儲存影片並顯示處理結果
        video_path = f"temp_video.{file_extension}"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        process_video(video_path)
