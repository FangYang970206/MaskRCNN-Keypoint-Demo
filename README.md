# MaskRCNN-Keypoint-Demo
This repo is about Mask RCNN with human-keypoint. More info can be seen in this [repo](https://github.com/chrispolo/Keypoints-of-humanpose-with-Mask-R-CNN).

#Requirements
* Python 3.4+
* TensorFlow 1.12
* keras 2.1.6
* numpy, skimage, scipy, Pillow, cython, h5py
* cocoapi: `pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"`
* cv2: `pip install opencv-python`

# How to run
Two choices:
```bash
python main.py --image path/to/image
```
or
```bash
python main.py --image path/to/video
```
The default choice is the `ski.jpg` in the `media/` folder.

# Result
![demo.gif](gif/demo.gif)