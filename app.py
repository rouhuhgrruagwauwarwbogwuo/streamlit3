import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
import cv2
import tempfile
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from mtcnn import MTCNN
from tensorflow.keras.optimizers import Adam

# ⬇️ 下載模型（如果還沒下載）
def download_model():
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    model_filename = "deepfake_cnn_model.h5"
    if not os.path.exists(model_filename):
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_filename, "wb") as f:
                f.write(response.content)
            print("模型下載成功！")
        else:
            print(f"下載失敗，狀態碼：{response.status_code}")
            return None
    return model_filename

# ✅ 載入 ResNet50 模型
try:
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    resnet_classifier = Sequential([
        resnet_model,
        Dense(1, activation='sigmoid')
    ])

    # 解凍 ResNet50 模型的最後幾層進行微調
    for layer in resnet_model.layers[:-4]:  # 保留 ResNet50 頂層，解凍其餘部分
        layer.trainable = False
    for layer in resnet_model.layers[-4:]:  # 微調最後幾層
        layer.trainable = True

    resnet_classifier.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    print("ResNet50 模型已載入並微調")
except Exception as e:
    print(f"ResNet50 載入錯誤：{e}")
    resnet_classifier = None

# ✅ MTCNN 初始化
detector = MTCNN()

# ✅ 擷取人臉
def extract_face(pil_img):
    img_array = np.array(pil_img)
    faces = detector.detect_faces(img_array)
    if len(faces) > 0:
        x, y, w, h = faces[0]['box']
        face = img_array[y:y+h, x:x+w]
        return Image.fromarray(face)
    return None

# ✅ 圖片處理方法
def apply_clahe(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def high_pass_filter(img):
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    return cv2.subtract(img, blurred)

# ✅ 中心裁切
def center_crop(img, target_size=(224, 224)):
    width, height = img.size
    new_width, new_height = target_size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    return img.crop((left, top, left + new_width, top + new_height))

# ✅ 預處理
def preprocess_for_resnet(pil_img):
    img = center_crop(pil_img, (224, 224))
    img_array = np.array(img)

    # ✅ 圖像增強
    img_array = apply_clahe(img_array)
    img_array = sharpen_image(img_array)
    img_array = high_pass_filter(img_array)

    img_array = img_array.astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# ✅ 預測
def predict_with_resnet(img):
    resnet_input = preprocess_for_resnet(img)
    pred = resnet_classifier.predict(resnet_input, verbose=0)[0][0]
    label = "Deepfake" if pred > 0.5 else "Real"
    return label, pred

# ✅ 顯示預測
def show_prediction(img):
    label, confidence = predict_with_resnet(img)
    st.image(img, caption="輸入圖片", use_container_width=True)
    st.subheader(f"ResNet50 判斷：**{label}**（信心度：{confidence:.2%}）")

# ✅ 資料增強設置（如果需要進行訓練）
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# ✅ Streamlit UI
st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖片偵測器")

tab1, tab2 = st.tabs(["🖼️ 圖片偵測", "🎥 影片偵測"])

# ✅ 圖片偵測
with tab1:
    st.header("上傳圖片進行 Deepfake 偵測")
    uploaded_image = st.file_uploader("請選擇圖片", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="原始圖片", use_container_width=True)

        face_img = extract_face(pil_img)
        if face_img:
            st.image(face_img, caption="偵測到人臉", width=300)
            show_prediction(face_img)
        else:
            st.info("⚠️ 未偵測到人臉，將使用整張圖片進行預測")
            show_prediction(pil_img)

# ✅ 影片偵測（僅擷取前幾幀）
with tab2:
    st.header("影片偵測（僅擷取前幾幀進行判斷）")
    uploaded_video = st.file_uploader("請上傳影片", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        import cv2
        st.info("🎬 擷取幀與分析中...")
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        shown = False

        while cap.isOpened() and not shown:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 10 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb)
                face_img = extract_face(pil_frame)
                if face_img:
                    st.image(face_img, caption=f"第 {frame_idx} 幀偵測到人臉", width=300)
                    show_prediction(face_img)
                    shown = True
            frame_idx += 1

        cap.release()
