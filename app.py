import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import load_model
from mtcnn import MTCNN
import tempfile
import os

# 載入模型
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# 偵測人臉
def extract_face(image):
    detector = MTCNN()
    image_np = np.array(image)
    results = detector.detect_faces(image_np)
    if results:
        x, y, w, h = results[0]['box']
        face = image_np[y:y+h, x:x+w]
        return Image.fromarray(face)
    else:
        return image

# 中心裁切
def center_crop(img, target_size):
    width, height = img.size
    new_width, new_height = target_size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return img.crop((left, top, right, bottom))

# 圖片預處理
def preprocess_image(img):
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img = center_crop(img, (224, 224))
    img_array = np.array(img)
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
    resnet_input = preprocess_input(np.expand_dims(img_array, axis=0))
    return resnet_input

# ResNet 預測
def predict_with_resnet(img):
    img_array = preprocess_image(img)
    prediction = resnet_model.predict(img_array)
    confidence = float(np.mean(prediction))
    label = "deepfake" if confidence >= 0.5 else "real"
    return label, confidence

# 頁面介面
st.title("🔍 Deepfake 偵測系統")
st.header("請上傳圖片或影片，我們將判斷其真實性")

uploaded_file = st.file_uploader("請選擇圖片或影片", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    file_type = uploaded_file.type
    
    if 'image' in file_type:
        img = Image.open(uploaded_file).convert('RGB')
        face_img = extract_face(img)
        st.image(face_img, caption='擷取的人臉', use_container_width=True)
        label, confidence = predict_with_resnet(face_img)
        st.subheader(f"🧠 判斷結果：{label.upper()} ({confidence:.2f})")

    elif 'video' in file_type:
        st.video(uploaded_file)
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 10 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb_frame)
                face_img = extract_face(img_pil)
                label, confidence = predict_with_resnet(face_img)
                st.image(face_img, caption=f"Frame {frame_count} - {label.upper()} ({confidence:.2f})", use_container_width=True)

            frame_count += 1

        cap.release()
        os.unlink(tfile.name)
