import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import face_recognition

# 🔹 載入 ResNet50 模型
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
resnet_classifier = Sequential([
    resnet_model,
    Dense(1, activation='sigmoid')  # 1 個輸出節點（0: 真實, 1: 假）
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔹 載入自訂 CNN 模型
custom_model = load_model('deepfake_cnn_model.h5')

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

# 🔹 人臉偵測
def extract_face(image_path):
    img = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(img)
    
    if len(face_locations) > 0:
        top, right, bottom, left = face_locations[0]
        face_img = img[top:bottom, left:right]
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
    face_img = extract_face(image_path)
    
    if face_img is not None:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # 轉為 RGB 格式
        resnet_label, resnet_confidence, custom_label, custom_confidence = predict_with_both_models(face_img)
    else:
        resnet_label, resnet_confidence, custom_label, custom_confidence = predict_with_both_models(image_path)
    
    # 顯示圖片
    img = image.load_img(image_path, target_size=(256, 256))
    plt.imshow(img)
    plt.axis('off')  # 隱藏座標軸
    
    # 顯示預測結果
    plt.title(f"ResNet50: {resnet_label} ({resnet_confidence:.2%})\n"
              f"Custom CNN: {custom_label} ({custom_confidence:.2%})")
    plt.show()

# 🔹 使用模型進行預測
image_path = 'test_image.jpg'  # 替換成你的測試圖片
show_prediction(image_path)     # 顯示圖片與預測結果
