import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from keras.applications import ResNet50, EfficientNetB0, Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
from mtcnn.mtcnn import MTCNN

st.set_page_config(page_title="Deepfake 偵測系統", layout="wide")

# 標題
st.title("🧠 Deepfake 偵測系統（多模型整合 + MTCNN）")

# 初始化 MTCNN 偵測器（加快效能可用 cache）
@st.cache_resource
def load_mtcnn():
    return MTCNN()

detector = load_mtcnn()

# 建立模型
def build_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

resnet_model = build_model(ResNet50(weights="imagenet", include_top=False))
efficientnet_model = build_model(EfficientNetB0(weights="imagenet", include_top=False))
xception_model = build_model(Xception(weights="imagenet", include_top=False))

# 人臉偵測（使用 MTCNN）
def extract_face_mtcnn(pil_img):
    img_np = np.array(pil_img)
    results = detector.detect_faces(img_np)
    if results:
        largest = max(results, key=lambda face: face["box"][2] * face["box"][3])
        x, y, w, h = largest["box"]
        x, y = max(x, 0), max(y, 0)
        face = img_np[y:y + h, x:x + w]
        return Image.fromarray(face)
    return None

# CLAHE 銳化
def apply_clahe_sharpening(img):
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_img, -1, sharpen_kernel)
    return sharpened

# 高通濾波
def apply_high_pass_filter(img):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = gray.shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 30
    mask[crow-r:crow+r, ccol-r:ccol+r] = 0
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back)

# 預測
def predict_with_ensemble(img):
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred1 = resnet_model.predict(img_array, verbose=0)[0][0]
    pred2 = efficientnet_model.predict(img_array, verbose=0)[0][0]
    pred3 = xception_model.predict(img_array, verbose=0)[0][0]

    avg_pred = (pred1 + pred2 + pred3) / 3
    label = "Deepfake" if avg_pred > 0.55 else "Real"
    return label, avg_pred, [pred1, pred2, pred3]

# 顯示信心分數條
def display_confidence_bar(preds):
    labels = ["ResNet50", "EfficientNetB0", "Xception"]
    colors = ["skyblue", "lightgreen", "salmon"]
    fig, ax = plt.subplots()
    ax.barh(labels, preds, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence (越接近1代表 Deepfake)")
    st.pyplot(fig)

# 使用者上傳
option = st.radio("選擇上傳類型", ("圖片", "影片"))

if option == "圖片":
    uploaded_file = st.file_uploader("請上傳圖片", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        pil_img = Image.open(uploaded_file).convert("RGB")
        face_img = extract_face_mtcnn(pil_img)
        if face_img is None:
            st.warning("⚠️ 無法偵測人臉，請確認圖片品質")
        else:
            clahe_img = apply_clahe_sharpening(face_img)
            hp_filtered_img = apply_high_pass_filter(clahe_img)
            final_img = Image.fromarray(hp_filtered_img).convert("RGB")

            st.image([pil_img, face_img, final_img], caption=["原始圖片", "偵測人臉", "處理後圖片"], width=250)

            label, avg_pred, model_preds = predict_with_ensemble(final_img)
            st.markdown(f"### 🧪 預測結果：**{label}**")
            st.write(f"平均信心分數：{avg_pred:.4f}")
            display_confidence_bar(model_preds)

elif option == "影片":
    uploaded_video = st.file_uploader("請上傳影片", type=["mp4", "mov", "avi"])
    if uploaded_video:
        with st.spinner("影片處理中..."):
            cap = cv2.VideoCapture(uploaded_video.name)
            frame_count = 0
            frame_confidences = []

            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= 100:  # 最多處理 100 幀
                    break
                if frame_count % 10 == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    face_img = extract_face_mtcnn(pil_img)
                    if face_img:
                        clahe_img = apply_clahe_sharpening(face_img)
                        hp_filtered_img = apply_high_pass_filter(clahe_img)
                        final_img = Image.fromarray(hp_filtered_img).convert("RGB")
                        label, avg_pred, _ = predict_with_ensemble(final_img)
                        frame_confidences.append((label, avg_pred))
                frame_count += 1
            cap.release()

            if not frame_confidences:
                st.warning("⚠️ 無法在影片中偵測到人臉")
            else:
                deepfake_count = sum(1 for label, _ in frame_confidences if label == "Deepfake")
                real_count = sum(1 for label, _ in frame_confidences if label == "Real")
                final_label = "Deepfake" if deepfake_count > real_count else "Real"
                avg_conf = np.mean([score for _, score in frame_confidences])

                st.markdown(f"### 🎥 影片判斷結果：**{final_label}**")
                st.write(f"平均信心分數：{avg_conf:.4f}")
