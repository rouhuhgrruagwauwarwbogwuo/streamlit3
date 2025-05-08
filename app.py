import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50, EfficientNetB0, Xception
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from retinaface import RetinaFace
import os
import tempfile
import gdown

# 設定標題
st.set_page_config(page_title="Deepfake 偵測系統")
st.title("🕵️ Deepfake 偵測系統")

# --------- 模型下載與載入（避免大檔錯誤） ---------
@st.cache_resource
def download_model(model_url, output_path):
    if not os.path.exists(output_path):
        gdown.download(model_url, output_path, quiet=False)
    return load_model(output_path)

# 替換為你自己的 Hugging Face 或 Google Drive 下載連結
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
effnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
xception_model = Xception(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# 你自己訓練的最終分類層模型（僅做分類）
custom_model_url = 'https://huggingface.co/yourname/yourmodel/resolve/main/custom_model.h5'
custom_model_path = 'custom_model.h5'
classifier_model = download_model(custom_model_url, custom_model_path)

# --------- 人臉擷取函式 ---------
def extract_face_fallback(img):
    img_np = np.array(img)
    faces = RetinaFace.detect_faces(img_np)
    if isinstance(faces, dict) and faces:
        first_key = list(faces.keys())[0]
        face_area = faces[first_key]["facial_area"]
        x1, y1, x2, y2 = face_area
        face = img_np[y1:y2, x1:x2]
    else:
        # OpenCV 備援人臉偵測
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces_cv = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces_cv) > 0:
            x, y, w, h = faces_cv[0]
            face = img_np[y:y+h, x:x+w]
        else:
            return img
    return Image.fromarray(face)

# --------- 圖像預處理 ---------
def preprocess_face(face_img, model_type='resnet'):
    img = face_img.resize((224, 224))
    img_np = np.array(img).astype('float32')
    if model_type == 'resnet':
        img_np = resnet_preprocess(img_np)
    elif model_type == 'efficientnet':
        img_np = effnet_preprocess(img_np)
    elif model_type == 'xception':
        img_np = xception_preprocess(img_np)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

# --------- 預測與集成 ---------
def predict_face(face_img):
    resnet_input = preprocess_face(face_img, 'resnet')
    effnet_input = preprocess_face(face_img, 'efficientnet')
    xception_input = preprocess_face(face_img, 'xception')

    resnet_feat = resnet_model.predict(resnet_input, verbose=0)
    effnet_feat = effnet_model.predict(effnet_input, verbose=0)
    xception_feat = xception_model.predict(xception_input, verbose=0)

    features = np.concatenate([resnet_feat, effnet_feat, xception_feat], axis=-1)
    prediction = classifier_model.predict(features, verbose=0)[0][0]

    label = "🟢 真實 (Real)" if prediction < 0.5 else "🔴 假的 (Deepfake)"
    confidence = 1 - prediction if prediction < 0.5 else prediction

    return label, float(confidence)

# --------- 使用者上傳圖片 ---------
uploaded_file = st.file_uploader("請上傳一張人臉圖片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="原始圖片", use_column_width=True)

    with st.spinner("正在擷取人臉並進行分析..."):
        face = extract_face_fallback(img)
        st.image(face, caption="擷取的人臉", use_column_width=True)

        result_label, result_score = predict_face(face)

        st.subheader("🔍 預測結果")
        st.markdown(f"**結果：{result_label}**")
        st.progress(result_score)

        st.markdown(f"信心分數：`{result_score:.2f}`")
