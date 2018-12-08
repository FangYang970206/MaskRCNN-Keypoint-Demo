# MaskRCNN-Keypoint-Demo
This repo is about Mask RCNN with human-keypoint. More info can be seen in this [repo](https://github.com/chrispolo/Keypoints-of-humanpose-with-Mask-R-CNN).

# Requirements
* Python 3.4+
* TensorFlow 1.3+
* keras 2.0.8
* numpy, skimage, scipy, Pillow, cython, h5py
* cocoapi: `pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"`
* cv2: `pip install opencv-python`

# How to run
first, clone this repo,
```bash
$ git clone https://github.com/FangYang970206/MaskRCNN-Keypoint-Demo
```
then,
```bash
$ cd MaskRCNN-Keypoint-Demo
```
download the **pre-trained model**([baiduyun](https://pan.baidu.com/s/19foQjAu3KSFsIooPuSLAbw), [google drive](https://drive.google.com/open?id=1NvayrcxR9v0kVeH-qRPfQR5mQlbOPQvt)) in the MaskRCNN-Keypoint-Demo folder.

finally, you have two choices:
```bash
$ python main.py --image path/to/image
```
or
```bash
$ python main.py --video path/to/video
```
Example:
```
$ python main.py --image media/ski.jpg
$ python main.py --video media/human.mp4
```
# Result
![demo.gif](gif/demo.gif)