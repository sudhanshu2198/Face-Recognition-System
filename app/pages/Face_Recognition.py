import streamlit as st
import requests
from PIL import Image
import numpy as np
import os
import cv2
from utils import visualization

st.title("Face Recognition")
st.success("Either select Image from Default Images or Upload Image")
uploaded_file = st.file_uploader(
    "Input Image File", type=["jpg", "png", "jpeg", "webp"]
)

options = [
    "Default_IMG_1.png",
    "Default_IMG_2.png",
    "Default_IMG_3.png",
    "Default_IMG_4.png",
    "Default_IMG_5.png",
    "Default_IMG_6.png",
    "Default_IMG_7.png",
    "Default_IMG_8.png",
    "Default_IMG_9.png",
]
selected_option = st.sidebar.selectbox("Choose a Image:", options)
button = st.button("Detect")

if button:
    if uploaded_file:
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post(
            "https://srastog-face-recognition-system.hf.space/predict", files=files
        )
        img = np.array(Image.open(uploaded_file))
    else:
        root_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        img_pth = os.path.join(root_dir, "data", "test_img_folder", selected_option)
        with open(img_pth, "rb") as image_file:
            files = {"file": (img_pth.split(os.path.sep)[-1], image_file.read())}
            response = requests.post(
                "https://srastog-face-recognition-system.hf.space/predict", files=files
            )
        img = cv2.imread(img_pth)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if response.status_code == 200:
        boxes = response.json()["boxes"]
        matrixs = response.json()["matrixs"]
        predictions = response.json()["predictions"]
        keypoints = response.json()["keypoints"]

        visualization(img, boxes, matrixs, keypoints, predictions)
    else:
        st.error("Upload failed!")
