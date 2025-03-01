import argparse
from image_tagger.utility import preprocess_img
from image_tagger.components import face_detection
from image_tagger.components import face_alignment
from image_tagger.components import face_recognition

def tagger(img):
	boxes,labels,scores=face_detection(img)
	aligned,matrixs,keypoints=face_alignment(img, boxes)
	results=face_recognition(aligned)

	return (boxes, matrixs, keypoints, results)

if  __name__ == '__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument("-p","--path",required=True)
	args=vars(parser.parse_args())

	img=preprocess_img(args["path"])
	boxes, matrixs, results = tagger(img)

