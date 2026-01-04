import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
import os

# Title and description
st.title("🎯 Military Object Detection System")
st.write("Upload an image or video to detect soldiers, weapons, vehicles, and more using YOLOv8.")

# Load YOLO model
@st.cache_resource
def load_model():
    model = YOLO("D:/Bhuvana/FinalProject/best.pt")
    return model

model = load_model()

# Sidebar for input options
option = st.sidebar.selectbox("Select input type", ("Image", "Video"))

# --------------------- IMAGE UPLOAD ---------------------
if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Run detection
        st.write("Detecting objects...")
        results = model.predict(image, conf=0.4)

        # Display result
        result_img = results[0].plot()  # Draw boxes and labels
        st.image(result_img, caption="Detection Result", use_column_width=True)

        # Option to download
        result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        temp_path = "detected_image.jpg"
        result_pil.save(temp_path)
        with open(temp_path, "rb") as f:
            st.download_button("Download Result", f, file_name="detected_image.jpg")

# --------------------- VIDEO UPLOAD ---------------------
elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        st.video(uploaded_video)

        st.write("Processing video, please wait...")

        cap = cv2.VideoCapture(tfile.name)
        out_file = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width, height = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        st.success("✅ Detection completed!")
        st.video(out_file)

        with open(out_file, "rb") as f:
            st.download_button("Download Processed Video", f, file_name="detected_video.mp4")
