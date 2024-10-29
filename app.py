import cv2
import streamlit as st
import yolov5
import os

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
        img = cv2.imread(uploaded_file.name)

        # Detect objects using the YOLO model
        results = model(img)

        # Display the image with detected objects
        st.image(results.render()[0], use_column_width=True)
elif mode == "Use Camera":
    # Use the camera for real-time object detection
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects using the YOLO model
        results = model(frame)

        # Display the frame with detected objects
        st.image(results.render()[0], use_column_width=True, channels="BGR")

        # Add a button to stop the camera
        if st.button("Stop"):
            break

    cap.release()

# Add a link to the GitHub repository
st.markdown(f"[View the code on GitHub](https://github.com/your-username/yolo-camera-app)")

# Add a link to the Hugging Face deployment
st.markdown(f"[Try the app on Hugging Face](https://huggingface.co/spaces/your-username/yolo-camera-app)")
