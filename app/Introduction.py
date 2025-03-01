import streamlit as st
import requests
from PIL import Image
import numpy as np
from utils import visualization

st.title("Image Tagger")
uploaded_file=st.file_uploader("Input Image File",type = ['jpg','png','jpeg','webp'])

if uploaded_file:
	files={"file":(uploaded_file.name,uploaded_file,uploaded_file.type)}
	response=requests.post("http://127.0.0.1:80/predict", files=files)
	if response.status_code == 200:
		boxes=response.json()["boxes"]
		matrixs=response.json()["matrixs"]
		predictions=response.json()["predictions"]
		keypoints=response.json()["keypoints"]

		img=np.array(Image.open(uploaded_file))
		visualization(img,boxes,matrixs,keypoints,predictions)
	else:
		st.error("Upload failed!")
	


