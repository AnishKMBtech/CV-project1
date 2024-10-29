import streamlit as st
import torch
from PIL import Image
import numpy as np

# Load the YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Streamlit app
st.title("YOLO Object Detection")

# Add option to upload an image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    img = Image.open(uploaded_file)

    # Detect objects using the YOLO model
    results = model(np.array(img))

    # Display the image with detected objects
    st.image(results.render()[0], use_column_width=True)

# Add a link to the GitHub repository
st.markdown(f"[View the code on GitHub](https://github.com/your-username/yolo-camera-app)")

# Add a link to the Hugging Face deployment
st.markdown(f"[Try the app on Hugging Face](https://huggingface.co/spaces/your-username/yolo-camera-app)")
