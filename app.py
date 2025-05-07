import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from PIL import Image, ImageEnhance, ImageFilter
import tempfile
import os
import requests
from mtcnn import MTCNN

# 🔽 下載自訂 CNN 模型（從 Hugging Face）
def download_model():
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    model_filename = "deepfake_cnn_model.h5"
    
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
try:
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    resnet_classifier = Sequential([
        resnet_model,
        Dense(1, activation='sigmoid')  
    ])
    resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("ResNet50 模型已成功載入")
except Exception as e:
    print(f"載入 ResNet50 模型時發生錯誤：{e}")
    resnet_classifier = None

# 🔹 載入自訂 CNN 模型
model_path = download_model()
if model_path:
    try:
        custom_model = load_model(model_path)
        print("自訂 CNN 模型已成功載入")
    except Exception as e:
        print(f"載入自訂 CNN 模型時發生錯誤：{e}")
        custom_model = None
else:
    custom_model = None

# 🔹 初始化 MTCNN 人臉檢測器
detector = MTCNN()

# 🔹 擷取圖片中的人臉
def extract_face(pil_img):
    img_array = np.array(pil_img)
    faces = detector.detect_faces(img_array)

    if len(faces) > 0:
        x, y, width, height = faces[0]['box']
        face = img_array[y:y+height, x:x+width]
        face_pil = Image.fromarray(face)
        return face_pil
    else:
        return None

# 🔹 中心裁切函數
def center_crop(img, target_size=(224, 224)):
    width, height = img.size
    new_width, new_height = target_size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    return img.crop((left, top, right, bottom))

# 🔹 CLAHE 預處理
def apply_clahe(image):
    enhancer = ImageEnhance.Contrast(image)
    image_clahe = enhancer.enhance(2)  # 增加對比度
    return image_clahe

# 🔹 頻域分析 (FFT)
def apply_fft(image):
    img_gray = image.convert('L')  # 轉換為灰階圖像
    img_array = np.array(img_gray)

    # 計算傅立葉變換
    f = np.fft.fft2(img_array)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)

    # 轉換回圖像
    magnitude_spectrum_img = Image.fromarray(np.uint8(magnitude_spectrum * 255 / magnitude_spectrum.max()))
    return magnitude_spectrum_img

# 🔹 圖片預處理（包括 CLAHE 和 FFT）
def preprocess_for_both_models(img):
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img = apply_clahe(img)  # CLAHE 處理
    img = apply_fft(img)    # 頻域處理
    img = center_crop(img, (224, 224))  # 中心裁切
    img_array = np.array(img)
    img_array = img_array.astype(np.float32) / 255.0

    # 擴展維度以符合模型輸入要求： (batch_size, height, width, channels)
    resnet_input = np.expand_dims(img_array, axis=0)
    custom_input = np.expand_dims(img_array, axis=0)

    return resnet_input, custom_input

# 🔹 模型預測
def predict_with_both_models(img, only_resnet=False):
    resnet_input, custom_input = preprocess_for_both_models(img)
    resnet_prediction = resnet_classifier.predict(resnet_input)[0][0]
    resnet_label = "Deepfake" if resnet_prediction > 0.5 else "Real"

    if only_resnet or not custom_model:
        return resnet_label, resnet_prediction, "", 0.0
    else:
        # 確保輸入的數據與模型預期的維度一致
        custom_prediction = custom_model.predict(custom_input)[0][0]
        custom_label = "Deepfake" if custom_prediction > 0.5 else "Real"
        return resnet_label, resnet_prediction, custom_label, custom_prediction

# 🔹 顯示預測結果
def show_prediction(img, only_resnet=False):
    resnet_label, resnet_confidence, custom_label, custom_confidence = predict_with_both_models(img, only_resnet)

    st.image(img, caption="原始圖片", use_container_width=True)
    st.image(img, caption="偵測到的人臉", use_container_width=False, width=300)
    st.subheader(f"ResNet50: {resnet_label} ({resnet_confidence:.2%})")
    if not only_resnet:
        st.subheader(f"Custom CNN: {custom_label} ({custom_confidence:.2%})")

# 🔹 Streamlit 主頁面
st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖片偵測器")

# ---------- 圖片 ---------- 
st.header("圖片偵測")
uploaded_image = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])
if uploaded_image:
    pil_img = Image.open(uploaded_image).convert("RGB")
    st.image(pil_img, caption="原始圖片", use_container_width=True)

    face_img = extract_face(pil_img)
    if face_img:
        st.image(face_img, caption="偵測到的人臉", use_container_width=False, width=300)
        show_prediction(face_img, only_resnet=True)
    else:
        st.write("未偵測到人臉，使用整體圖片進行預測")
        show_prediction(pil_img, only_resnet=True)
