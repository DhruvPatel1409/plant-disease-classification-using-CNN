import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Load the model and class indices
model = tf.keras.models.load_model('plantvillage_model.h5')
class_indices = json.load(open('class_indices.json'))

def load_and_preprocess_image(image_path, target_size=(224,224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_image = load_and_preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]  # Convert index to string
    return predicted_class_name

st.title("🌿 Plant Disease Detection 🌱")
uploaded_image = st.file_uploader("Upload an image ...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    
    col1, col2 = st.columns(2)
    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)
    with col2:
        if st.button("CLASSIFY"):
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f"Prediction: {str(prediction)}")
