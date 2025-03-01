import json
import dlib
import numpy as np
import onnxruntime as ort
from image_tagger import config
from image_tagger.utility import FaceAligner
from image_tagger.utility import face_similarity


def face_recognition(aligned):
    with open(config.FACE_EMBEDDING, 'r') as file:
        actor_embeddings=json.load(file)
    
    face_identified=[]
    for face in aligned:
        embedding=face_embedding(face)
        identity=[]
        for actor in sorted(actor_embeddings.keys()):
            similarity=face_similarity(embedding,actor_embeddings[actor])
            identity.append((actor,similarity))
        identity=[n for i,(n,similar) in enumerate(sorted(identity,key=lambda x:x[1])) if i<3]
        face_identified.append(identity)
    return face_identified

def face_detection(img):
    session=ort.InferenceSession(config.FACE_DETECTION_ONNX_WEIGHTS, 
                                 providers=['CPUExecutionProvider'])
    output=session.run([], {'input':img[np.newaxis,...]})
    return output

def face_alignment(img, boxes):
    predictor=dlib.shape_predictor(config.FACE_KEYPOINT_DLIB_WEIGHTS)
    aligner=FaceAligner(predictor)
    matrix=[]
    aligned=[]
   
    for i,box in enumerate(boxes):
        align_image,align_matrix=aligner.align(img,box)
        matrix.append(align_matrix)
        aligned.append(align_image)
        
    return (aligned,matrix)
    
def face_embedding(img):
    img=(img/255.0-config.MEAN)/config.STD
    img=np.transpose(img,(2,0,1)).astype(np.float32)
    img=img[np.newaxis,...]
    session=ort.InferenceSession(config.FACE_EMBEDDING_ONNX_WEIGHTS)
    output=session.run([], {'input':img})[0][0].tolist()
    return output