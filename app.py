import streamlit as st
import cv2
import yolov5

# Load the YOLO model
model = yolov5.load('yolov5s')

# Create the Streamlit app
st.title("YOLO Object Detection")

# Add file uploader to allow users to upload an image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image
    img = cv2.imread(uploaded_file.name)

    # Detect objects using the YOLO model
    results = model(img)

    # Display the image with detected objects
    st.image(results.render()[0], use_column_width=True)
