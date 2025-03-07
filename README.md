# Introduction

![](https://raw.githubusercontent.com/sudhanshu2198/Face-Recognition-System/main/assests/images/face_recognition.png)

This project focuses on developing a Face Recognition System for identifying Bollywood celebrities using deep learning techniques. The system can detect, recognize, and classify faces of popular Bollywood actors from images or videos. By leveraging Convolutional Neural Networks (CNNs) and Siamese Networks, the model learns unique facial features and accurately matches them against a pre-trained database of Bollywood stars.

The project involves key steps such as **Face Detection → Landmark Detection → Face Alignment → Face Embedding Generation → Face Recognition** for accurate identification. Potential applications include celebrity identification in media, automated tagging in photos, and AI-powered fan engagement tools.

## Project Links:
1. [Streamlit Webapp](https://bollywood-celebrities-face-recognition-system.streamlit.app/)
2. [FastAPI Backend](https://srastog-face-recognition-system.hf.space/docs)
3. [Documentation](https://sudhanshu2198.github.io/Face-Recognition-System/)

## Resources

| No | Description            | Dataset | Preprocessing | Modelling | Weights | ONNX Conversion |
|:---| :-----------------: | :-----: | :--------:    | :-------: | :-----: | :-----: |
|1| Face Detection | [Link](https://www.kaggle.com/datasets/sudhanshu2198/human-face-detection-dataset)|  [Link](https://www.kaggle.com/code/sudhanshu2198/human-face-detection-data)        |   [Link](https://www.kaggle.com/code/sudhanshu2198/face-detection-fasterrcnn-mobilenet-model)     |   [Link](https://www.kaggle.com/models/sudhanshu2198/fasterrcnn-mobilenet)     |   [Link](https://www.kaggle.com/code/sudhanshu2198/fasterrcnn-model-onnx-conversion)     |
|2| Face Keypoint Detection | [Link](https://www.kaggle.com/datasets/sudhanshu2198/face-keypoint-detection-data)|  [Link](https://www.kaggle.com/code/sudhanshu2198/keypoints-detection-dataset)        |   [Link](https://www.kaggle.com/code/sudhanshu2198/human-face-keypoint-detection)     |   [Link](https://www.kaggle.com/models/sudhanshu2198/facial-keypoint-detection-model)     |   [Link](https://www.kaggle.com/code/sudhanshu2198/keypoint-model-onnx-conversion)     |
|3| Face Embedding - Siamese | [Link](https://www.kaggle.com/datasets/sudhanshu2198/indian-celebtities-face-recognition)|  [Link](https://www.kaggle.com/code/sudhanshu2198/indian-celebrities-face-extraction-alignment)        |   [Link](https://www.kaggle.com/code/sudhanshu2198/indian-celebrities-face-recognition)     |   [Link](https://www.kaggle.com/models/sudhanshu2198/face-recognition)     |   [Link](https://www.kaggle.com/code/sudhanshu2198/face-recognition-onxx-conversion)     |

## 🛠 Skills
Pytorch, Torchvision, Ultralytics, OpenCV, Dlib,  Numpy, Streamlit, FastAPI, Git 

## Directory Tree
```bash

├── api
│   │── __init__.py
│   └── main.py
├── app
│   ├── Introduction.py
│   ├── __init__.py
│   ├── utils.py
│   ├── pages
│   │   └── Face_Recognition.py
│   ├── assests
│   │   ├── images
|   │   │   ├── face_alignment.png
|   │   │   ├── face_detection.png
|   │   │   ├── face_embedding.png
|   │   │   └──  face_recognition.png
│   │   ├── videos
|   │   │   └── working_demo.mp4
│   ├── data
│   │   ├── actor_embedding.json
│   │   ├── test_img_folder
|   │   │   ├── DEFAULT_IMG_1.png
|   │   │   ├── DEFAULT_IMG_2.png
|   │   │   └──  ...
│   ├── docs
│   │   ├── face_alignment.md
│   │   ├── face_detection.md
│   │   ├── face_recognition.md
│   │   ├── face_embedding.md
│   │   └── index.md
│   ├── src
│   │   ├── image_tagger
|   │   │   ├── __init__.py
|   │   │   ├── components.py
|   │   │   ├── config.py
|   │   │   ├── main.py
|   │   │   └── utility.py
│   ├── weights
│   │   ├── face_detection_onnx_32.onnx
│   │   ├── keypoint_68_weights.dat
│   │   └── siamese_onnx_32.onnx
│── .gitignore
│── .pre-commit-config.yaml
│── README.md
│── mkdocs.yml
│── pyproject.toml
└── requirements.txt
```
## Run Webapp Locally

Clone the project

```bash
  git clone https://github.com/sudhanshu2198/Face-Recognition-System
```

Change to project directory

```bash
  cd Face-Recognition-System
```
Create Virtaul Environment and install dependencies

```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
```

Run FastAPI Backend
```bash

  uvicorn api.main:app --host 0.0.0.0 --port 80 --reload
  

  ```
Run Streamlit App
```bash

  streamlit run Introduction.py
  

  ```


