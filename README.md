# Introduction

<video controls>
<source src="https://raw.githubusercontent.com/sudhanshu2198/Face-Recognition-System/main/assests/videos/working_demo.mp4" type="video/mp4">
</video>

This project focuses on developing a Face Recognition System for identifying Bollywood celebrities using deep learning techniques. The system can detect, recognize, and classify faces of popular Bollywood actors from images or videos. By leveraging Convolutional Neural Networks (CNNs) and Siamese Networks, the model learns unique facial features and accurately matches them against a pre-trained database of Bollywood stars.

The project involves key steps such as **Face Detection → Landmark Detection → Face Alignment → Face Embedding Generation → Face Recognition** for accurate identification. Potential applications include celebrity identification in media, automated tagging in photos, and AI-powered fan engagement tools.

## Project Links:
1. [Streamlit Webapp](https://bollywood-celebrities-face-recognition-system.streamlit.app/)
2. [FastAPI Backend](https://srastog-face-recognition-system.hf.space/docs)
3. [Github Repo](https://github.com/sudhanshu2198/Face-Recognition-System)

## Resources

| No | Description            | Dataset | Preprocessing | Modelling | Weights |
|:---| :-----------------: | :-----: | :--------:    | :-------: | :-----: |
|1| Face Detection | [Link](https://www.kaggle.com/datasets/sudhanshu2198/human-face-detection-dataset)|  [Link](https://www.kaggle.com/code/sudhanshu2198/human-face-detection-data)        |   [Link](https://www.kaggle.com/code/sudhanshu2198/face-detection-fasterrcnn-mobilenet-model)     |   [Link](https://www.kaggle.com/models/sudhanshu2198/fasterrcnn-mobilenet)     |
|2| Face Keypoint Detection | [Link](https://www.kaggle.com/datasets/sudhanshu2198/face-keypoint-detection-data)|  [Link](https://www.kaggle.com/code/sudhanshu2198/keypoints-detection-dataset)        |   [Link](https://www.kaggle.com/code/sudhanshu2198/human-face-keypoint-detection)     |   [Link](https://www.kaggle.com/models/sudhanshu2198/facial-keypoint-detection-model)     |


## Face Detection
![]("https://raw.githubusercontent.com/sudhanshu2198/Face-Recognition-System/main/assests/images/face_detection.png")
- The first step in face recognition is detecting the face in an image or video.
- It involves identifying the location of faces and drawing bounding boxes around them.
- Faster RCNN Model is used for face detection.

## Face Alignment
![]("https://raw.githubusercontent.com/sudhanshu2198/Face-Recognition-System/main/assests/images/face_alignment.png")
- Once a face is detected, key landmarks on the face (e.g., eyes, nose, mouth, jawline) are identified.
- These keypoints help in further processing such as alignment and feature extraction.
- Face alignment uses the detected keypoints for aligning eyes along a horizontal axis

## Face Recognition
![]("https://raw.githubusercontent.com/sudhanshu2198/Face-Recognition-System/main/assests/images/face_embedding.png")
- After aligning the face, Siamese network extracts a feature vector that uniquely represents the face.
- The generated embeddings are compared with stored embeddings in a database using similarity metrics like Euclidean Distance.
- Based on the similarity score, the system verifies whether two images belong to the same person.""")

