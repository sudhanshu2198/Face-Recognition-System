import numpy as np
import os

PROJECT_DIR=os.path.dirname(os.getcwd())
WEIGHT_DIR=os.path.join(PROJECT_DIR,"weights")
DATA_DIR=os.path.join(PROJECT_DIR,"data")
FACE_DETECTION_ONNX_WEIGHTS=os.path.join(WEIGHT_DIR,"face_detection_onnx_32.onnx")
FACE_KEYPOINT_DLIB_WEIGHTS=os.path.join(WEIGHT_DIR,"keypoint_68_weights.dat")
FACE_EMBEDDING_ONNX_WEIGHTS=os.path.join(WEIGHT_DIR,"siamese_onnx_32.onnx")
FACE_EMBEDDING=os.path.join(DATA_DIR,"actor_embedding.json")
MEAN=np.array([0.485,0.456,0.406])
STD=np.array([0.229,0.224,0.225])
THRESHOLD=10.50
PADDING=15
SIZE=224