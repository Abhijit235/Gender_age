import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import os
os.system("bash setup.sh")


# Load the saved model
model = load_model('age_gender_model.keras')


# Streamlit App
st.title("Age and Gender Detection App")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the uploaded image
    image = Image.open(uploaded_file)
    image = np.array(image)
    image_resized = cv2.resize(image, (128, 128)) / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension


    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict age and gender
    age_pred, gender_pred = model.predict(image_resized)
    age = int(age_pred[0][0])
    # Adjusting the threshold for gender classification
    gender = 'Male' if gender_pred[0][0] > 0.1 else 'Female'



    # Display the results
    st.write(f"Predicted Age: {age}")
    st.write(f"Gender raw prediction: {gender}")

