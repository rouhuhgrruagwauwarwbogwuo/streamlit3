import streamlit as st
import numpy as np
import cv2
from mtcnn import MTCNN
import tempfile
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense
import requests
import os
from PIL import Image

st.set_page_config(page_title="Deepfake 偵測系統", layout="wide")

# ====== 下載與載入模型 ======
@st.cache_resource
def load_models():
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    model_path = "/tmp/deepfake_cnn_model.h5"
    if not os.path.exists(model_path):
        response = requests.get(model_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)

    custom_model = load_model(model_path)

    resnet_base = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
    resnet_classifier = Sequential([
        resnet_base,
        Dense(1, activation='sigmoid')
    ])
    resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return custom_model, resnet_classifier

custom_model, resnet_classifier = load_models()
detector = MTCNN()

# ====== 預處理與人臉擷取 ======
def preprocess_image_cv2(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_gray = clahe.apply(img_gray)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    return cv2.resize(img_rgb, (256, 256)) / 255.0

def extract_face(image):
    faces = detector.detect_faces(image)
    if faces:
        x, y, w, h = faces[0]['box']
        return image[y:y+h, x:x+w]
    return None

# ====== 模型預測 ======
def predict_frame(img):
    if img.shape[:2] != (256, 256):
        img = cv2.resize(img, (256, 256))
    img_resnet = preprocess_input(np.expand_dims(img.copy(), axis=0))
    img_custom = np.expand_dims(img / 255.0, axis=0)

    resnet_pred = resnet_classifier.predict(img_resnet, verbose=0)[0][0]
    custom_pred = custom_model.predict(img_custom, verbose=0)[0][0]
    return resnet_pred, custom_pred

# ====== 圖片處理 ======
def handle_image_upload(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    face = extract_face(img_np)
    processed = preprocess_image_cv2(face if face is not None else img_np)

    resnet_pred, custom_pred = predict_frame(processed)
    resnet_label = "偽造" if resnet_pred > 0.5 else "真實"
    custom_label = "偽造" if custom_pred > 0.5 else "真實"

    st.image(img, caption="上傳圖片", use_container_width=True)
    st.markdown(f"""
    ### 🔍 預測結果
    - **ResNet50 模型**：{resnet_label}（信心值：{resnet_pred:.2%}）
    - **自訂 CNN 模型**：{custom_label}（信心值：{custom_pred:.2%}）
    """)

# ====== 影片處理 ======
def handle_video_upload(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    predictions = []
    stframe = st.empty()

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if i % 10 != 0:
            continue  # 每 10 幀處理一次

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = extract_face(rgb_frame)
        processed = preprocess_image_cv2(face if face is not None else rgb_frame)
        resnet_pred, custom_pred = predict_frame(processed)

        label = "偽造" if resnet_pred > 0.5 else "真實"
        cv2.putText(frame, f"ResNet50: {label} ({resnet_pred:.2%})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        stframe.image(frame, channels="BGR", use_container_width=True)

        predictions.append({
            "影格編號": i,
            "ResNet50 信心值": f"{resnet_pred:.2%}",
            "自訂模型 信心值": f"{custom_pred:.2%}"
        })

    cap.release()
    return predictions

# ====== 主介面 ======
st.title("🧠 Deepfake 偵測系統")
st.sidebar.title("📂 載入媒體")
option = st.sidebar.radio("選擇偵測類型", ("圖片", "影片"))

if option == "圖片":
    uploaded_image = st.sidebar.file_uploader("請上傳圖片（JPG/PNG）", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        handle_image_upload(uploaded_image)

elif option == "影片":
    uploaded_video = st.sidebar.file_uploader("請上傳影片（MP4/AVI）", type=["mp4", "avi", "mov"])
    if uploaded_video:
        with st.spinner("正在處理影片...請稍候"):
            preds = handle_video_upload(uploaded_video)

        st.success("✅ 影片處理完成！")
        st.write("### 偵測結果表格")
        st.dataframe(preds)
