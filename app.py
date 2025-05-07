import streamlit as st
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from PIL import Image
import cv2
from mtcnn import MTCNN

# Load ResNet50 model
resnet_model = ResNet50(weights='imagenet')

def extract_face(pil_img):
    detector = MTCNN()
    img_array = np.array(pil_img.convert("RGB"))
    faces = detector.detect_faces(img_array)

    if faces:
        x, y, width, height = faces[0]['box']
        x, y = max(0, x), max(0, y)
        face = img_array[y:y+height, x:x+width]
        face = cv2.resize(face, (224, 224))
        return Image.fromarray(face)
    else:
        return pil_img.resize((224, 224))

def predict_with_resnet(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = resnet_model.predict(img_array)
    top_prediction = decode_predictions(predictions, top=1)[0][0]
    label, confidence = top_prediction[1], float(top_prediction[2])

    # Simplified decision logic for Deepfake vs Real
    if confidence > 0.5:
        if label.lower() in ["mask", "wig", "stage", "screen", "robot"]:
            return "Deepfake", confidence * 100
        else:
            return "Real", confidence * 100
    else:
        return "Uncertain", confidence * 100

# Streamlit UI
st.title("Deepfake Detection with ResNet50")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)
    st.image(pil_img, caption='Uploaded Image', use_container_width=True)

    face_img = extract_face(pil_img)
    label, confidence = predict_with_resnet(face_img)

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}%")

    if label == "Uncertain":
        st.warning("The model is not confident enough to classify this image as Real or Deepfake.")
