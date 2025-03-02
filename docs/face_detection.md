# Face Detection

## Faster R-CNN

Faster R-CNN (Region-based Convolutional Neural Network) is a deep learning model designed for object detection, which identifies objects in an image and classifies them into categories.
How It Works

- Feature Extraction - A convolutional neural network (CNN) extracts feature maps from the input image.
- Region Proposal Network (RPN) - Generates region proposals where objects might be located.
- ROI Pooling - Extracts fixed-size feature maps from proposed regions.
- Classification & Bounding Box Regression - Classifies each region and refines bounding box coordinates.

## Loss Function

1. Classification Loss (Cross-Entropy Loss): Ensures that the predicted class labels match the ground truth labels.

2. Bounding Box Regression Loss (Smooth L1 Loss): Measures the accuracy of predicted bounding box coordinates compared to ground truth.

## Non-Maximum Suppression (NMS)

Non-Maximum Suppression (NMS) is a post-processing technique used to eliminate redundant bounding boxes detected around the same object. NMS ensures that only the most relevant bounding box is retained for each detected object, improving detection accuracy and reducing false positives.
It works as follows:

- Sort detected bounding boxes by confidence scores.

- Select the highest confidence box and remove overlapping boxes with Intersection over Union (IoU) greater than a threshold.

- Repeat until no boxes remain.

## Further Reading
- **[Training Faster RCNN on Custom Dataset](https://debuggercafe.com/how-to-train-faster-rcnn-resnet50-fpn-v2-on-custom-dataset/)**
- **[Mean Average Precision Metric](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)**
- **[Non Max Suppression](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/)**