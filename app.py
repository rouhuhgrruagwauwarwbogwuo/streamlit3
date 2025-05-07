import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
import requests
from io import BytesIO
from mtcnn import MTCNN
import dlib

# 設定頁面配置
st.set_page_config(page_title="Deepfake 偵測", layout="centered")
st.title("Deepfake 偵測工具")

# 載入 ResNet50 預訓練模型
resnet_model = ResNet50(weights='imagenet')

# 載入 MTCNN 面部偵測模型
mtcnn = MTCNN()

# 載入 Dlib 預訓練的人臉特徵點偵測模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 需要下載 Dlib 預訓練模型

# 圖像中央補白補滿
def center_crop(img, target_size=(224, 224)):
    width, height = img.size
    new_width, new_height = target_size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return img.crop((left, top, right, bottom))

# 面部偵測與區域擷取
def detect_face(img):
    img_array = np.array(img)
    faces = mtcnn.detect_faces(img_array)
    if len(faces) > 0:
        x, y, width, height = faces[0]['box']
        face_img = img_array[y:y+height, x:x+width]
        return face_img
    else:
        return None

# 使用 Dlib 特徵點偵測
def detect_landmarks(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) > 0:
        shape = predictor(gray, faces[0])
        return shape
    return None

# 對圖片執行 FFT 與高速遮漏
def apply_fft_high_pass(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows // 2 , cols // 2
    fshift[crow-20:crow+20, ccol-20:ccol+20] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_back

# Unsharp Mask 提升詳細
def apply_unsharp_mask(img):
    gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
    unsharp = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    return unsharp

# 合併預處理 (FFT + USM)
def preprocess_advanced(img):
    # 大幅縮放 保護詳細
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img = center_crop(img, (224, 224))
    img_array = np.array(img)

    # FFT 高速遮漏
    high_pass_img = apply_fft_high_pass(img_array)
    high_pass_img_color = cv2.merge([high_pass_img]*3)  # 換 RGB

    # Unsharp Mask
    enhanced_img = apply_unsharp_mask(high_pass_img_color)

    # 接合所有這些元素 (最終對上 ResNet50)
    final_input = preprocess_input(np.expand_dims(enhanced_img, axis=0))

    return final_input, enhanced_img

# ResNet50 預測
def predict_with_resnet(img_tensor):
    predictions = resnet_model.predict(img_tensor)
    decoded = decode_predictions(predictions, top=3)[0]
    label = decoded[0][1]
    confidence = float(decoded[0][2])
    return label, confidence, decoded

# 載入自訂模型
def load_custom_model_from_huggingface(model_url):
    response = requests.get(model_url)
    
    # 檢查下載是否成功
    if response.status_code == 200:
        try:
            # 下載模型後，從 bytes 載入模型
            model = load_model(BytesIO(response.content))
            print("模型成功載入")
            return model
        except Exception as e:
            print(f"載入模型失敗: {e}")
            return None
    else:
        print(f"下載失敗，HTTP 狀態碼: {response.status_code}")
        return None

# 自訂模型預測
def predict_with_custom_model(img_tensor):
    predictions = custom_model.predict(img_tensor)
    confidence = predictions[0][0]  # 假設是二分類模型，返回預測信心度
    return confidence

# 載入自訂模型
custom_model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
custom_model = load_custom_model_from_huggingface(custom_model_url)

if custom_model is None:
    print("自訂模型加載失敗，請檢查模型文件或 URL。")
else:
    print("自訂模型已加載")

# 偵測預測結果
uploaded_file = st.file_uploader("上傳影像", type=["jpg", "png", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    img_tensor, enhanced_img = preprocess_advanced(img)
    
    # 使用ResNet50進行預測
    label, confidence, decoded_predictions = predict_with_resnet(img_tensor)
    st.image(enhanced_img, caption="處理後影像", use_container_width=True)
    st.write(f"ResNet50 預測結果: {label} (信心分數: {confidence:.2f})")
    
    # 自訂模型預測
    custom_confidence = predict_with_custom_model(img_tensor)
    st.write(f"自訂模型預測信心分數: {custom_confidence:.2f}")
    
    # 顯示結果
    if custom_confidence > 0.5:
        st.write("這是一張 Deepfake 影像")
    else:
        st.write("這是一張真實影像")
