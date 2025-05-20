import streamlit as st
import numpy as np
import os
import cv2
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate, Input
from tensorflow.keras.applications.efficientnet import preprocess_input
import mediapipe as mp

st.set_page_config(page_title="Deepfake 偵測器（多分支分析）", layout="wide")
st.title("🧠 Deepfake 圖像偵測器（模仿人類注意行為）")

# 載入 EfficientNet 基礎模型
def build_branch_model():
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base.output)
    return Model(inputs=base.input, outputs=x)

# 建構多分支模型
@st.cache_resource
def build_multibranch_model():
    full_img_input = Input(shape=(224, 224, 3))
    eye_input = Input(shape=(224, 224, 3))
    mouth_input = Input(shape=(224, 224, 3))

    full_branch = build_branch_model()(full_img_input)
    eye_branch = build_branch_model()(eye_input)
    mouth_branch = build_branch_model()(mouth_input)

    merged = Concatenate()([full_branch, eye_branch, mouth_branch])
    output = Dense(1, activation='sigmoid')(merged)
    model = Model(inputs=[full_img_input, eye_input, mouth_input], outputs=output)
    return model

model = build_multibranch_model()

# 擷取臉部、眼睛、嘴巴區域
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True)

def extract_landmark_regions(pil_img):
    img_np = np.array(pil_img)
    results = face_mesh.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    h, w, _ = img_np.shape
    
    if not results.multi_face_landmarks:
        return None, None, None

    landmarks = results.multi_face_landmarks[0].landmark
    
    def get_bbox(indices, margin=10):
        points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        xs, ys = zip(*points)
        x1, y1 = max(min(xs) - margin, 0), max(min(ys) - margin, 0)
        x2, y2 = min(max(xs) + margin, w), min(max(ys) + margin, h)
        return x1, y1, x2, y2

    eye_idx = list(range(33, 133)) + list(range(133, 144))
    mouth_idx = list(range(78, 88)) + list(range(88, 95)) + list(range(95, 105))

    x1, y1, x2, y2 = get_bbox(range(len(landmarks)))
    fx = pil_img.crop((x1, y1, x2, y2)).resize((224, 224))

    ex1, ey1, ex2, ey2 = get_bbox(eye_idx)
    eye = pil_img.crop((ex1, ey1, ex2, ey2)).resize((224, 224))

    mx1, my1, mx2, my2 = get_bbox(mouth_idx)
    mouth = pil_img.crop((mx1, my1, mx2, my2)).resize((224, 224))

    return fx, eye, mouth

# 預測
@st.cache_resource
def load_weights(model):
    # 請根據需要替換為實際訓練後的模型權重載入
    # model.load_weights("your_model_weights.h5")
    return model

model = load_weights(model)

def predict_image(img):
    full_img, eye_img, mouth_img = extract_landmark_regions(img)
    if not all([full_img, eye_img, mouth_img]):
        return "無法偵測關鍵區域", 0.0

    def preprocess(pil):
        arr = np.array(pil).astype(np.float32)
        arr = preprocess_input(arr)
        return arr

    inputs = [
        np.expand_dims(preprocess(full_img), axis=0),
        np.expand_dims(preprocess(eye_img), axis=0),
        np.expand_dims(preprocess(mouth_img), axis=0)
    ]
    pred = model.predict(inputs, verbose=0)[0][0]
    label = "Deepfake" if pred > 0.5 else "Real"
    return label, pred

# 視覺化結果
def show_result(img):
    st.image(img, caption="輸入圖像", use_container_width=True)
    label, conf = predict_image(img)
    st.subheader(f"預測結果：**{label}**")
    st.markdown(f"信心分數：**{conf:.2f}**")

    fig, ax = plt.subplots(figsize=(6, 1))
    ax.barh([0], conf, color='green' if label == "Real" else 'red')
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('信心分數')
    st.pyplot(fig)

# 主介面
tab1, tab2 = st.tabs(["🖼️ 圖像偵測", "🧪 模型說明"])

with tab1:
    st.header("上傳圖像進行 Deepfake 偵測（多分支注意分析）")
    uploaded = st.file_uploader("選擇一張圖片", type=["jpg", "jpeg", "png"])
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        show_result(pil_img)

with tab2:
    st.markdown("""
    ### 模型說明
    - 使用 EfficientNetB0 為骨幹模型進行特徵提取。
    - 模仿人類注意策略：
        - 全臉掃描
        - 針對眼睛與嘴唇區域進行詳細分析
    - 三分支合併後進行分類判斷。
    """)
