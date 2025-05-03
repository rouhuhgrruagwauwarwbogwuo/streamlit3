import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- 圖像預處理方法 ---
def preprocess_image(img, method="none", target_size=(256, 256)):
    try:
        img = img.resize(target_size)
        img_array = image.img_to_array(img).astype('uint8')

        if method == "none":
            pass  # 無處理

        elif method == "clahe":
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            img_array = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

        elif method == "sharpen":
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            img_array = cv2.filter2D(img_array, -1, kernel)

        elif method == "clahe_sharpen":
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            img_array = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            img_array = cv2.filter2D(img_array, -1, kernel)

        img_array = img_array / 255.0  # normalize
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"預處理錯誤：{e}")
        return None

# --- 模型預測 ---
def predict_image(model, img, method="none"):
    preprocessed = preprocess_image(img, method)
    if preprocessed is None:
        return None, None
    prediction = model.predict(preprocessed)[0][0]
    label = "Deepfake" if prediction > 0.5 else "Real"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

# --- Streamlit App 主程式 ---
st.set_page_config(page_title="Deepfake 圖片偵測", layout="centered")
st.title("🧠 Deepfake 圖片偵測")

# 模型載入
@st.cache_resource
def load_deepfake_model():
    return load_model("deepfake_cnn_model.h5")

model = load_deepfake_model()

# 上傳圖片
uploaded_file = st.file_uploader("請上傳圖片", type=["jpg", "jpeg", "png"])

# 預處理選單
method = st.selectbox("選擇預處理方式", ["none", "clahe", "sharpen", "clahe_sharpen"], index=0)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="上傳的圖片", use_column_width=True)

    if st.button("🔍 進行預測"):
        label, confidence = predict_image(model, img, method)

        if label:
            st.markdown(f"### 🧾 預測結果：`{label}`")
            st.markdown(f"### ✅ 信心分數：`{confidence:.2%}`")
        else:
            st.error("預測失敗，請確認圖片是否正確。")
