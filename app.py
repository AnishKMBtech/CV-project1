import streamlit as st
import yolov5
import numpy as np
from PIL import Image

# Load the YOLO model
model = yolov5.load('yolov5s')

# Streamlit app
st.title("YOLO Object Detection")

# Add option to upload an image or use the camera
mode = st.radio("Choose mode:", options=["Upload Image", "Use Camera"])

if mode == "Upload Image":
    # Allow user to upload an image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Read the uploaded image
        img = Image.open(uploaded_file)

        # Detect objects using the YOLO model
        results = model(np.array(img))

        # Display the image with detected objects
        st.image(results.render()[0], use_column_width=True)
elif mode == "Use Camera":
    st.write("Unfortunately, the camera feature is not available in this version of the app. Please use the 'Upload Image' mode.")

# Add a link to the GitHub repository
st.markdown(f"[View the code on GitHub](https://github.com/your-username/yolo-camera-app)")

# Add a link to the Hugging Face deployment
st.markdown(f"[Try the app on Hugging Face](https://huggingface.co/spaces/your-username/yolo-camera-app)")
