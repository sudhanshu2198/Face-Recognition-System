import streamlit as st
import os

st.title("Project Description")

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
video_path = os.path.join(root_dir, "assests", "videos", "working_demo.mp4")
video_file = open(video_path, "rb")
video_bytes = video_file.read()

st.video(video_bytes)
st.write(
    "This project focuses on developing a Face Recognition System for identifying Bollywood celebrities using deep learning techniques. The system can detect, recognize, and classify faces of popular Bollywood actors from images or videos. By leveraging Convolutional Neural Networks (CNNs) and Siamese Networks, the model learns unique facial features and accurately matches them against a pre-trained database of Bollywood stars."
)
st.markdown(
    "**Face Detection → Landmark Detection → Face Alignment → Face Embedding Generation → Face Recognition**"
)

st.header("Face Detection")
st.image(
    "https://raw.githubusercontent.com/sudhanshu2198/Face-Recognition-System/main/assests/images/face_detection.png"
)
st.markdown("""
	        - The first step in face recognition is detecting the face in an image or video.
	        - It involves identifying the location of faces and drawing bounding boxes around them.
	        - Faster RCNN Model is used for face detection.
	       """)

st.header("Face Alignment")
st.image(
    "https://raw.githubusercontent.com/sudhanshu2198/Face-Recognition-System/main/assests/images/face_alignment.png"
)
st.markdown("""
	        - Once a face is detected, key landmarks on the face (e.g., eyes, nose, mouth, jawline) are identified.
	        - These keypoints help in further processing such as alignment and feature extraction.
	        - Face alignment uses the detected keypoints for aligning eyes along a horizontal axis
	        """)

st.header("Face Recognition")
st.image(
    "https://raw.githubusercontent.com/sudhanshu2198/Face-Recognition-System/main/assests/images/face_embedding.png"
)
st.markdown("""
	        - After aligning the face, Siamese network extracts a feature vector that uniquely represents the face.
	        - The generated embeddings are compared with stored embeddings in a database using similarity metrics like Euclidean Distance.
	        - Based on the similarity score, the system verifies whether two images belong to the same person.""")

st.header("Project Links")
st.markdown(
    "- **[Documentation Link](%s)**"
    % "https://sudhanshu2198.github.io/Face-Recognition-System/"
)
st.markdown(
    "- **[Github Repo Link](%s)**"
    % "https://github.com/sudhanshu2198/Face-Recognition-System"
)
st.markdown(
    "- **[FastAPI Backend Link](%s)**"
    % "https://sudhanshu2198.github.io/Face-Recognition-System/"
)
