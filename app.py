import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from io import BytesIO
from mtcnn import MTCNN

# 設定頁面配置
st.set_page_config(page_title="Deepfake 偵測", layout="centered")
st.title("Deepfake 偵測工具")

# 載入 ResNet50 預訓練模型
resnet_model = ResNet50(weights='imagenet')

# 載入 MTCNN 面部偵測模型
mtcnn = MTCNN()

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

# 預處理影像，確保符合模型要求的格式
def preprocess_advanced(img):
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img = center_crop(img, (224, 224))
    img_array = np.array(img)
    
    # 確保 img_array 是 float32 且標準化
    img_array = img_array.astype(np.float32)
    img_array /= 255.0

    # 增加批次維度 (batch dimension)
    img_array = np.expand_dims(img_array, axis=0)
    
    return preprocess_input(img_array), img

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
    
    if response.status_code == 200:
        try:
            # 從 bytes 載入模型
            model = load_model(BytesIO(response.content))
            return model
        except Exception as e:
            print(f"載入模型失敗: {e}")
            return None
    else:
        print(f"下載失敗，HTTP 狀態碼: {response.status_code}")
        return None

# 自訂模型預測
def predict_with_custom_model(img_tensor):
    if custom_model is None:
        print("自訂模型尚未加載，請檢查模型加載過程。")
        return None
    
    # 預測
    predictions = custom_model.predict(img_tensor)
    
    # 假設是二分類模型，返回預測信心度
    confidence = predictions[0][0]
    return confidence

# 載入自訂模型
custom_model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
custom_model = load_custom_model_from_huggingface(custom_model_url)

# 偵測結果
uploaded_file = st.file_uploader("上傳影像", type=["jpg", "png", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    img_tensor, original_img = preprocess_advanced(img)
    
    # 顯示原始圖片
    st.image(original_img, caption="原始圖片", use_container_width=True)
    
    # 使用ResNet50進行預測
    label, confidence, decoded_predictions = predict_with_resnet(img_tensor)
    st.write(f"ResNet50 預測結果: {label} (信心分數: {confidence:.2f})")
    
    # 自訂模型預測
    custom_confidence = predict_with_custom_model(img_tensor)
    if custom_confidence is not None:
        st.write(f"自訂模型預測信心分數: {custom_confidence:.2f}")
    
    # 顯示結果
    if custom_confidence is not None and custom_confidence > 0.5:
        st.write("這是一張 Deepfake 影像")
    else:
        st.write("這是一張真實影像")
