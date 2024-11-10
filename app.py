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
uploaded_file = st.file_uploader("Choose an image", type="jpg")

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128, 128))  # Resize to input size
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Display image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Predict age and gender
    predictions = model.predict(img_array)
    predicted_age = predictions[0]
    predicted_gender = 'Female' if predictions[1] > 0.5 else 'Male'

  
    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict age and gender
    age_pred, gender_pred = model.predict(image_resized)
    age = int(age_pred[0][0])
    # Adjusting the threshold for gender classification
    gender = 'Male' if gender_pred[0][0] < 0.1 else 'Female'



    # Display the results
    st.write(f"Predicted Age: {predicted_age}")
    st.write(f"Predicted Gender: {predicted_gender}")

