import os
import sys
import random
import argparse
import numpy as np
import cv2 as cv

import coco
import utils
import model as modellib


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    KEYPOINT_MASK_POOL_SIZE = 7


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--image", type=str)
    parse.add_argument('--video', type=str)
    args = parse.parse_args()

    ROOT_DIR = os.getcwd()

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    if not os.path.exists(COCO_MODEL_PATH):
        raise AssertionError('please download the pre-trained model')

    colorsFile = "colors.txt"
    with open(colorsFile, 'rt') as f:
        colorsStr = f.read().rstrip('\n').split('\n')
    colors = []
    for i in range(len(colorsStr)):
        rgb = colorsStr[i].split(' ')
        color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
        colors.append(color)

    inference_config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=inference_config)

    model.load_weights(COCO_MODEL_PATH, by_name=True)

    if (args.image):
        if not os.path.isfile(args.image):
            print("Input image file ", args.image, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.image)
        outputFile = args.image[:-4]+'_mask_rcnn_out_py.jpg'
    elif (args.video):
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.video)
        outputFile = args.video[:-4]+'_mask_rcnn_out_py.avi'
    else:
        cap = cv.VideoCapture(0)

    if (not args.image):
        vid_writer = cv.VideoWriter(outputFile,
                                    cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                    30,
                                    (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                                     round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    maskThreshold = 0.3
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            break
        
        print("frame shape:", frame.shape)
        # class_names = ['BG', 'person']
        results = model.detect_keypoint([frame], verbose=1)
        r = results[0]
        if r['masks'].shape[0]:
            for i in range(r['masks'].shape[2]):
                mask = r['masks'][:, :, i]
                mask = (mask > maskThreshold)
                roi = frame[mask]
                colorIndex = random.randint(0, len(colors)-1)
                color = colors[colorIndex]
                frame[mask] = ([0.3 * color[0],
                                0.3 * color[1],
                                0.3 * color[2]] + 0.7 * roi).astype(np.uint8)
                mask = mask.astype(np.uint8)
                _, contours, hierarchy = cv.findContours(mask,
                                                         cv.RETR_TREE,
                                                         cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(frame, contours, -1, color, 3,
                                cv.LINE_8, hierarchy, 100)
            keypoints = np.array(r['keypoints']).astype(int)
            skeleton = [0, -1, -1, 5, -1, 6, 5, 7, 6, 8, 7, 9,
                        8, 10, 11, 13, 12, 14, 13, 15, 14, 16]
            for i in range(len(keypoints)):
                # Skeleton: 11*2
                limb_colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170],
                               [0, 255, 0], [170, 255, 0], [255, 170, 0],
                               [255, 0, 0], [255, 0, 170], [170, 0, 255],
                               [170, 170, 0], [170, 0, 170]]
                if(len(skeleton)):
                    skeleton = np.reshape(skeleton, (-1, 2))
                    neck = np.array((keypoints[i, 5, :]
                                    + keypoints[i, 6, :]) / 2).astype(int)
                    if(keypoints[i, 5, 2] == 0 or keypoints[i, 6, 2] == 0):
                        neck = [0, 0, 0]
                    limb_index = -1
                    for limb in skeleton:
                        limb_index += 1
                        start_index, end_index = limb
                        if(start_index == -1):
                            Joint_start = neck
                        else:
                            Joint_start = keypoints[i][start_index]
                        if(end_index == -1):
                            Joint_end = neck
                        else:
                            Joint_end = keypoints[i][end_index]
                        if ((Joint_start[2] != 0) & (Joint_end[2] != 0)):
                            cv.line(frame,
                                    tuple(Joint_start[:2]),
                                    tuple(Joint_end[:2]),
                                    limb_colors[limb_index], 5)
        if (args.image):
            cv.imwrite(outputFile, frame.astype(np.uint8))
        else:
            vid_writer.write(frame.astype(np.uint8))


if __name__ == "__main__":
    main()
