import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import random

# 🔹 預處理圖片，確保 ResNet 和 自訂 CNN 都能處理
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

# 🔹 使用 ResNet50 模型進行預測
def predict_with_resnet(img):
    # 預處理圖片
    resnet_input, _ = preprocess_for_both_models(img)

    # 載入 ResNet50 模型
    resnet_model = ResNet50(weights='imagenet')

    # 預測結果
    predictions = resnet_model.predict(resnet_input)

    # 將預測結果轉為標籤與信心分數
    label = np.argmax(predictions)
    confidence = predictions[0][label]

    return label, confidence

# 🔹 影片偵測：逐幀處理影片並顯示結果
def process_video(input_video_path, output_video_path):
    # 讀取影片
    cap = cv2.VideoCapture(input_video_path)

    # 影片編碼器設置
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (640, 480))

    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        # 將當前幀轉為 PIL 影像格式
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 使用 ResNet50 進行預測
        label, confidence = predict_with_resnet(img)

        # 顯示結果文字
        label_text = 'Deepfake' if label == 1 else 'Real'
        confidence_text = f'{confidence * 100:.2f}%'

        # 在畫面上繪製結果
        cv2.putText(frame, f'{label_text} - {confidence_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 顯示處理後的幀
        out.write(frame)  # 保存處理過的幀
        cv2.imshow('frame', frame)  # 顯示當前幀

        # 若按下 'q' 鍵，退出處理
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 🔹 測試影片偵測
if __name__ == "__main__":
    input_video_path = "input_video.mp4"  # 請替換為輸入影片檔案路徑
    output_video_path = "output_video.avi"  # 請替換為輸出影片檔案路徑

    # 處理影片並保存結果
    process_video(input_video_path, output_video_path)
