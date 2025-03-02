# Face Alignment 

<img src="https://raw.githubusercontent.com/sudhanshu2198/Face-Recognition-System/main/assests/images/face_alignment.png">

Face alignment is a preprocessing step in face recognition and analysis that ensures faces are properly oriented by detecting key landmarks and adjusting the facial position accordingly. By aligning faces before processing, models can achieve more robust and consistent results in tasks like face recognition and analysis.

## Keypoint Detection
Dlib's 68 Face Keypoint Detector provides precise landmark points, which are used to align faces.
The key steps include:

- The model identifies 68 key points on the face, including the eyes, nose, mouth, and jawline.
- Using the eye positions, the tilt or rotation of the face is estimated.
- The image is rotated and scaled to align the eyes and mouth in a predefined standard position.

## Further Reading
- **[Face Alignment Explanation](https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/)**
- **[Face Keypoint Detection using Dlib](https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/)**

