import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import requests
import os

# Define model path
MODEL_PATH = "best.pt"

# GitHub Raw URL of the model file
MODEL_URL = "https://raw.githubusercontent.com/Nishanth1812/YOLO-Streamlit-App/main/best.pt"

# Function to download the model from GitHub
def download_model():
    with open(MODEL_PATH, "wb") as f:
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
            st.success("Model downloaded successfully!")
        else:
            st.error("Failed to download the model. Check the URL or repo permissions.")

# Download model if not present
if not os.path.exists(MODEL_PATH):
    st.write("Downloading YOLO model from GitHub...")
    download_model()

# Load YOLO model
model = YOLO(MODEL_PATH)

# Streamlit UI
st.title("YOLO Object Detection App")

# Upload image option
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Brightness slider
brightness = st.slider("Adjust Brightness", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

if uploaded_file:
    # Read image
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Adjust brightness
    image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)

    # Run YOLO model on the image
    results = model(image, conf=0.8)
    output_image = results[0].plot()

    # Convert frame to RGB for Streamlit
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(output_image_rgb)

    # Display the image
    st.image(img_pil, caption="Detected Objects", use_column_width=True)
