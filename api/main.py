import sys
import os
from PIL import Image
import numpy as np
from fastapi import FastAPI, UploadFile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from image_tagger.main import tagger

app=FastAPI(title="Image Tagger")

@app.get("/")
def display():
    return "Welcome to Image Tagger Api"

@app.post("/predict")
def predict(file: UploadFile):
    img=Image.open(file.file)
    img=np.array(img)
    img=(np.transpose(img,(2,0,1)).astype(dtype=np.float32))
    img/=255.0
    boxes, matrixs, results = tagger(img)

    return {"prediction":results,
            "boxes":boxes.tolist(),
            "matrix":[matrix.tolist() for matrix in matrixs]}