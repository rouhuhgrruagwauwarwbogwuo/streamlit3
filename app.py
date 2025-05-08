import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, Xception, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.xception import preprocess_input as preprocess_xception
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from PIL import Image
import cv2

# 預處理圖像 (強化預處理)
def apply_clahe_sharpen(img):
    img = np.array(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # 銳化處理
    kernel = np.array([[-1, -1, -1], [-1, 9,-1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)
    return Image.fromarray(img)

def high_pass_filter(img):
    img = np.array(img, dtype=np.float32)
    dft = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # 高通濾波器
    rows, cols = img.shape[:2]
    crow, ccol = rows // 2 , cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 30
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
    mask[mask_area] = 0
    
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = np.uint8(img_back)
    
    return Image.fromarray(img_back)

# 數據增強
def augment_image_v2(img):
    datagen = ImageDataGenerator(
        rotation_range=40,  
        width_shift_range=0.3,  
        height_shift_range=0.3,  
        shear_range=0.3,  
        zoom_range=0.3,  
        horizontal_flip=True,
        vertical_flip=True,  
        fill_mode='nearest'
    )
    
    img_array = np.array(img).reshape((1, ) + np.array(img).shape)
    augmented_img = next(datagen.flow(img_array, batch_size=1))
    return Image.fromarray(augmented_img[0].astype(np.uint8))

# 載入微調後的模型
def load_advanced_models():
    resnet_model = load_model("fine_tuned_resnet50.h5")
    efficientnet_model = load_model("fine_tuned_efficientnet.h5")
    xception_model = load_model("fine_tuned_xception.h5")
    
    return {
        'ResNet50': resnet_model,
        'EfficientNet': efficientnet_model,
        'Xception': xception_model
    }

# 預處理圖像
def preprocess_image_v2(img, model_name):
    img = augment_image_v2(img)  
    img = apply_clahe_sharpen(img)
    img = high_pass_filter(img)  

    if model_name == 'Xception':
        img = img.resize((299, 299))
        img_array = np.array(img).astype(np.float32)
        return preprocess_xception(img_array)
    else:
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32)
        if model_name == 'ResNet50':
            return preprocess_resnet(img_array)
        elif model_name == 'EfficientNet':
            return preprocess_efficientnet(img_array)
    return img_array

# 模型預測
def predict_model(models, img):
    predictions = []
    for model_name, model in models.items():
        img_array = preprocess_image_v2(img, model_name)
        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array)
        predictions.append(preds[0][0])  # 只取出第一個結果
    return predictions

# 集成模型預測
def stacking_predict_v2(models, img, threshold=0.7):  
    preds = predict_model(models, img)
    avg = np.mean(preds)
    label = "Deepfake" if avg > threshold else "Real"
    return label, avg

# 顯示預測結果
def show_prediction_v2(img, models, threshold=0.7):
    label, confidence = stacking_predict_v2(models, img, threshold)
    st.image(img, caption="輸入圖像", use_container_width=True)
    st.subheader(f"預測結果：**{label}**")
    st.markdown(f"信心分數：**{confidence:.2f}**")

    # 顯示信心分數條
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.barh([0], confidence, color='green' if label == "Real" else 'red')
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('信心分數')
    st.pyplot(fig)

# 主函數，Streamlit 介面
def main():
    st.title("Deepfake 偵測器")

    uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)

        # 加載預訓練模型
        models = load_advanced_models()

        # 顯示預測結果
        show_prediction_v2(img, models)

if __name__ == "__main__":
    main()
