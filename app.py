import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load YOLO model
model = YOLO(r"C:\Users\Devab\OneDrive\Desktop\Coding\ML-DL\Deep Learning\Applications\YOLO Finetuning\best.pt")

# Streamlit UI
st.title("YOLO Object Detection App (Webcam)")

# Brightness slider
brightness = st.slider("Adjust Brightness", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

start_button = st.button("Start Webcam")
stop_button = st.button("Stop Webcam")

# Initialize webcam
if start_button:
    web_cam = cv2.VideoCapture(0)
    web_cam.set(3, 640)
    web_cam.set(4, 480)

    frame_placeholder = st.empty()

    while web_cam.isOpened():
        success, img_frame = web_cam.read()
        if not success:
            break  # Exit if frame not captured properly

        # Adjust brightness
        img_frame = cv2.convertScaleAbs(img_frame, alpha=brightness, beta=0)

        # Run YOLO model on the frame
        results = model(img_frame, conf=0.8)
        a_frame = results[0].plot()

        # Convert frame to RGB for Streamlit
        output_frame_rgb = cv2.cvtColor(a_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(output_frame_rgb)

        # Display the frame in Streamlit
        frame_placeholder.image(img_pil, caption="Webcam Feed", use_column_width=True)

        if stop_button:
            break

    web_cam.release()
    cv2.destroyAllWindows()
