import streamlit as st
import cv2
import numpy as np

def visualization(img,boxes,matrixs,keypoints,predictions):
	st.subheader("Uploaded Image")
	st.image(img)
	
	non_aligned_images=[]
	aligned_images=[]
	keypoints_images=[]
	c1,c2,c3,c4=st.columns(4)
	with c1:
		st.markdown("**Face Detection**")
	with c2:
		st.markdown("**Face Keypoints**")
	with c3:
		st.markdown("**Face Alignment**")
	with c4:
		st.markdown("**Face Recognition**")

	for i in range(len(predictions)):
		box=boxes[i]
		box=list(map(int,box))
		matrix=np.array(matrixs[i])
		keypoint=np.array(keypoints[i])
		kp_array=list(zip(keypoint[:,0],keypoint[:,1]))
		keypoint=[cv2.KeyPoint(x=float(x), y=float(y), size=2.5) for x, y in kp_array]

		non_align_face=img[box[1]:box[3],box[0]:box[2]]
		non_align_face=cv2.resize(non_align_face,(224,224))

		align_face=cv2.warpAffine(img.copy(),matrix,(224,224),flags=cv2.INTER_CUBIC)

		keypoints_image=cv2.drawKeypoints(img,keypoint,0, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		keypoints_image=keypoints_image[box[1]:box[3],box[0]:box[2]]
		keypoints_image=cv2.resize(keypoints_image,(224,224))

		non_aligned_images.append(non_align_face)
		aligned_images.append(align_face)
		keypoints_images.append(keypoints_image)

	for i in range(len(predictions)):
		c1,c2,c3,c4=st.columns(4)
		with c1:
			st.image(non_aligned_images[i])
		with c2:
			st.image(keypoints_images[i])
		with c3:
			st.image(aligned_images[i])
		with c4:
			st.write([predict.replace("_"," ").capitalize()for predict in predictions[i]])