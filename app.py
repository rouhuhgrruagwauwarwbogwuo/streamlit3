import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import random

# 🔹 增加圖片增強與處理
def preprocess_for_both_models(img):
    # 1️⃣ **高清圖處理：LANCZOS 縮圖**
    img = img.resize((256, 256), Image.Resampling.LANCZOS)

    # 2️⃣ **ResNet50 必須 224x224**
    img = center_crop(img, (224, 224))

    img_array = np.array(img)  # 轉為 numpy array

    # 3️⃣ **可選：對 ResNet50 做 Gaussian Blur**
    apply_blur = True  # 🚀 這裡可以開關
    if apply_blur:
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)

    # 4️⃣ **進行顏色空間轉換：轉換到 YCbCr 或 Lab 空間**
    apply_color_space_conversion = True
    if apply_color_space_conversion:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)  # 轉換為 YCbCr 色彩空間
        # img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2Lab)  # 可選：轉換為 Lab 色彩空間

    # 5️⃣ **應用 CLAHE（對比度受限自適應直方圖均衡化）**
    apply_clahe = True
    if apply_clahe:
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2Lab)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_array = cv2.cvtColor(limg, cv2.COLOR_Lab2RGB)

    # 6️⃣ **隨機旋轉或縮放圖片（數據增強）**
    random_flip = random.choice([True, False])
    if random_flip:
        img_array = np.fliplr(img_array)  # 隨機左右翻轉

    random_rotate = random.choice([True, False])
    if random_rotate:
        angle = random.randint(-10, 10)
        height, width = img_array.shape[:2]
        matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
        img_array = cv2.warpAffine(img_array, matrix, (width, height))

    # 7️⃣ **ResNet50 特定預處理**
    resnet_input = preprocess_input(np.expand_dims(img_array, axis=0))

    # 8️⃣ **自訂 CNN 正規化 (0~1)**
    custom_input = np.expand_dims(img_array / 255.0, axis=0)

    return resnet_input, custom_input

# 🔹 中心裁剪函數：對圖片進行裁剪，使其符合目標大小
def center_crop(img, target_size):
    width, height = img.size
    target_width, target_height = target_size

    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = (width + target_width) // 2
    bottom = (height + target_height) // 2

    img = img.crop((left, top, right, bottom))
    return img

# 🔹 測試用的圖片處理示範
if __name__ == "__main__":
    # 載入圖片
    img_path = "your_image_path.jpg"  # 請替換為圖片路徑
    img = Image.open(img_path)

    # 預處理圖片，適用於 ResNet50 和自訂 CNN
    resnet_input, custom_input = preprocess_for_both_models(img)

    # 輸出處理後的圖片尺寸與格式
    print(f"ResNet50 Input Shape: {resnet_input.shape}")
    print(f"Custom CNN Input Shape: {custom_input.shape}")
