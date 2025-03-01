import cv2
import numpy as np
import dlib
from image_tagger import config

def preprocess_img(img_pth):
    img=cv2.imread(img_pth)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
    img=(np.transpose(img,(2,0,1)).astype(dtype=np.float32))
    img/=255.0
    
    return img

def face_similarity(embed_1,embed_2):
    return np.sqrt(np.sum(np.square(np.array(embed_1)-np.array(embed_2))))

class FaceAligner():
    def __init__(self,predictor,desired_left_eye=(0.35,0.35),desired_face_width=config.SIZE, desired_face_height=None):
        self.predictor=predictor
        self.desired_left_eye=desired_left_eye
        self.desired_right_eye=tuple([1-val for val in desired_left_eye])
        self.desired_face_width=desired_face_width
        self.desired_face_height=desired_face_height
        
        if self.desired_face_height is None:
            self.desired_face_height=self.desired_face_width
            
    def shape_to_np(self,shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def align(self,img,box):
        image=np.transpose(img,(1,2,0)).copy()
        image=(image*255).astype("uint8")

        x1=max(int(box[0])-config.PADDING,0)
        x2=min(int(box[2])+config.PADDING,image.shape[1])
        y1=max(int(box[1])-config.PADDING,0)
        y2=min(int(box[3])+config.PADDING,image.shape[0])
        
        dlib_box=dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
    
        face_keypoint=self.predictor(image,dlib_box)
        face_keypoint=self.shape_to_np(face_keypoint)
        
        left_eye_center=(np.mean(face_keypoint[36:42,0]),np.mean(face_keypoint[36:42,1]))
        right_eye_center=(np.mean(face_keypoint[42:48,0]),np.mean(face_keypoint[42:48,1]))

        dy=right_eye_center[1]-left_eye_center[1]
        dx=right_eye_center[0]-left_eye_center[0]
        
        angle=np.degrees(np.arctan2(dy, dx))

        dist=np.sqrt((dx ** 2) + (dy ** 2))
        desired_dist=self.desired_right_eye[0]-self.desired_left_eye[0]
        desired_dist*=self.desired_face_width
        scale=desired_dist/dist

        eye_center=((left_eye_center[0] + right_eye_center[0]) // 2,
                    (left_eye_center[1] + right_eye_center[1]) // 2)
      
        Matrix = cv2.getRotationMatrix2D(eye_center, angle, scale)
        
        tX = self.desired_face_width * 0.5
        tY = tY = self.desired_face_height * self.desired_left_eye[1]
        Matrix[0, 2] += (tX - eye_center[0])
        Matrix[1, 2] += (tY - eye_center[1])
        
        (w, h) = (self.desired_face_width, self.desired_face_height)
        aligned_img = cv2.warpAffine(image, Matrix, (w,h),flags=cv2.INTER_CUBIC)
        
        return (aligned_img, Matrix, face_keypoint)