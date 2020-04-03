# This script is used to estimate an accuracy of different face detection models.
# COCO evaluation tool is used to compute an accuracy metrics (Average Precision).
# Script works with different face detection datasets.
import json
from math import pi
import cv2 as cv
import argparse
import os
import sys
import dlib
import json
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

parser = argparse.ArgumentParser(
        description='Evaluate OpenCV face detection algorithms '
                    'using COCO evaluation tool, http://cocodataset.org/#cv_detections-eval')
parser.add_argument('--cv_proto', help='Path to .prototxt of Caffe model or .pbtxt of TensorFlow graph')
parser.add_argument('--cv_model', help='Path to .caffemodel trained in Caffe or .pb from TensorFlow')
parser.add_argument('--dlib_model', help='Path to .dat file for dlib')
parser.add_argument('--ann', help='Path to text file with ground truth annotations')
parser.add_argument('--pics', help='Path to images root directory')
parser.add_argument('--fddb', help='Evaluate FDDB dataset, http://vis-www.cs.umass.edu/fddb/', action='store_true')
args = parser.parse_args()

dataset = {}
dataset['images'] = []
dataset['categories'] = [{ 'id': 0, 'name': 'face' }]
dataset['annotations'] = []

def ellipse2Rect(params):
    rad_x = params[0]
    rad_y = params[1]
    angle = params[2] * 180.0 / pi
    center_x = params[3]
    center_y = params[4]
    pts = cv.ellipse2Poly((int(center_x), int(center_y)), (int(rad_x), int(rad_y)),
                          int(angle), 0, 360, 10)
    rect = cv.boundingRect(pts)
    left = rect[0]
    top = rect[1]
    right = rect[0] + rect[2]
    bottom = rect[1] + rect[3]
    return left, top, right, bottom

def addImage(imagePath):
    assert('images' in  dataset)
    imageId = len(dataset['images'])
    dataset['images'].append({
        'id': int(imageId),
        'file_name': imagePath
    })
    return imageId

def addBBox(imageId, left, top, width, height):
    assert('annotations' in  dataset)
    dataset['annotations'].append({
        'id': len(dataset['annotations']),
        'image_id': int(imageId),
        'category_id': 0,  # Face
        'bbox': [int(left), int(top), int(width), int(height)],
        'iscrowd': 0,
        'area': float(width * height)
    })

def addDetection(detections, imageId, left, top, width, height, score):
    detections.append({
      'image_id': int(imageId),
      'category_id': 0,  # Face
      'bbox': [int(left), int(top), int(width), int(height)],
      'score': float(score)
    })
    return detections

def fddb_dataset(annotations, images):
    d = 'FDDB-fold-05-ellipseList.txt'
    with open(os.path.join(annotations, d), 'rt') as f:
        lines = [line.rstrip('\n') for line in f]
        lineId = 0
        while lineId < len(lines):
            # Image
            imgPath = lines[lineId]
            lineId += 1
            imageId = addImage(os.path.join(images, imgPath) + '.jpg')

            img = cv.imread(os.path.join(images, imgPath) + '.jpg')

            # Faces
            numFaces = int(lines[lineId])
            lineId += 1
            for i in range(numFaces):
                params = [float(v) for v in lines[lineId].split()]
                lineId += 1
                left, top, right, bottom = ellipse2Rect(params)
                addBBox(imageId, left, top, width=right - left + 1,
                        height=bottom - top + 1)


def evaluate(gt_file, predictions_file):
    cocoGt = COCO(gt_file)
    cocoDt = cocoGt.loadRes(predictions_file)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats


### Convert to COCO annotations format #########################################
fddb_dataset(args.ann, args.pics)

with open('annotations.json', 'wt') as f:
    json.dump(dataset, f)

### Obtain a detection from OpenCV model ##########################################################
cv_detections = []
if args.cv_proto and args.cv_model:
    net = cv.dnn.readNet(args.cv_proto, args.cv_model)

    def detect_cv(img, imageId):
        global cv_detections
        imgWidth = img.shape[1]
        imgHeight = img.shape[0]
        net.setInput(cv.dnn.blobFromImage(img, 1.0, (300, 300), (104., 177., 123.), False, False))
        out = net.forward()

        for i in range(out.shape[2]):
            confidence = out[0, 0, i, 2]
            left = int(out[0, 0, i, 3] * img.shape[1])
            top = int(out[0, 0, i, 4] * img.shape[0])
            right = int(out[0, 0, i, 5] * img.shape[1])
            bottom = int(out[0, 0, i, 6] * img.shape[0])

            x = max(0, min(left, img.shape[1] - 1))
            y = max(0, min(top, img.shape[0] - 1))
            w = max(0, min(right - x + 1, img.shape[1] - x))
            h = max(0, min(bottom - y + 1, img.shape[0] - y))

            cv_detections = addDetection(cv_detections, imageId, x, y, w, h, score=confidence)

### Obtain a detection from dlib model ##########################################################
dlib_detections = []
if args.cv_proto and args.cv_model:
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.dlib_model)

    def detect_dlib(img, imageId):
        global dlib_detections
        dets = cnn_face_detector(img, 1)

        print("Number of faces detected: {}".format(len(dets)))
        for i, d in enumerate(dets):
            x = max(0, min(d.rect.left(), img.shape[1] - 1))
            y = max(0, min(d.rect.top(), img.shape[0] - 1))
            w = max(0, min(d.rect.right() - x + 1, img.shape[1] - x))
            h = max(0, min(d.rect.bottom() - y + 1, img.shape[0] - y))
            dlib_detections = addDetection(dlib_detections, imageId, x, y, w, h, score=d.confidence)

### Get detections for a dataset  #########################################

for i in range(len(dataset['images'])):
    sys.stdout.write('\r%d / %d' % (i + 1, len(dataset['images'])))
    sys.stdout.flush()

    img_for_cv = cv.imread(dataset['images'][i]['file_name'])
    img_for_dlib = dlib.load_rgb_image(dataset['images'][i]['file_name'])
    imageId = int(dataset['images'][i]['id'])

    detect_cv(img_for_cv, imageId)
    detect_dlib(img_for_dlib, imageId)

with open('cv_detections.json', 'wt') as f:
    json.dump(cv_detections, f)

with open('dlib_detections.json', 'wt') as f:
    json.dump(dlib_detections, f)

print("AP scores for OpenCV face detector\n")
evaluate('annotations.json', 'cv_detections.json')

print("AP scores for dlib face detector\n")
evaluate('annotations.json', 'dlib_detections.json')

### Split detections into several files by different thresholds #########################################

confidence_thresholds = np.linspace(.05, 0.9, int(0.9/0.05), endpoint=True)
for det_name in ['cv_detections', 'dlib_detections']:
    with open(det_name+".json", "r") as read_file:
        cv_detections = json.load(read_file)

    cv_detections_new = {}
    for threshold in confidence_thresholds:
        cv_detections_new[threshold] = []

    for detection in cv_detections:
        for threshold in confidence_thresholds:
            if detection["score"] >= threshold:
                cv_detections_new[threshold].append(detection)

    for threshold in confidence_thresholds:
        with open(det_name+'/'+det_name+'_'+str(round(threshold, 2))+'.json', 'wt') as f:
            json.dump(cv_detections_new[threshold], f)

### Get evaluations for all the files #######################################
aps = {}
for det_name in ['cv_detections', 'dlib_detections']:
    aps[det_name] = []
    for threshold in confidence_thresholds:
        stats = evaluate('annotations.json', det_name+'/'+det_name+'_'+str(round(threshold, 2))+'.json')
        aps[det_name].append(stats[1]) # the first value is for IoU=0.50, area=all, maxDets=100

### Visualize a dependency #######################################

plt.ylim(0, 1)
plt.xlim(0, 1)
plt.plot(confidence_thresholds, aps['cv_detections'])
plt.xlabel('confidence threshold')
plt.ylabel('AP value')
plt.show()

plt.ylim(0, 1)
plt.xlim(0, 1)
plt.plot(confidence_thresholds, aps['dlib_detections'])
plt.xlabel('confidence threshold')
plt.ylabel('AP value')
plt.show()

def rm(f):
    if os.path.exists(f):
        os.remove(f)
