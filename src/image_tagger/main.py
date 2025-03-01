import argparse
from image_tagger.utility import preprocess_img
from image_tagger.components import face_detection
from image_tagger.components import face_alignment
from image_tagger.components import face_recognition

def tagger(img_pth):
	img=preprocess_img(img_pth)
	boxes,labels,scores=face_detection(img)
	aligned,matrix=face_alignment(img, boxes)
	results=face_recognition(aligned)

	return (boxes, matrix, results)

if  __name__ == '__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument("-p","--path",required=True)
	args=vars(parser.parse_args())

	boxes, matrix, results = tagger(args["path"])
	print(type(boxes))
	print(type(matrix))
	print(results)

