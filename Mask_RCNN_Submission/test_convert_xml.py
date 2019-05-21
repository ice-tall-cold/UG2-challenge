import time
import os
import sys
import random
import math
import numpy as np
import skimage.io
import xml.etree.ElementTree as ET
import matplotlib
import matplotlib.pyplot as plt
import argparse

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library|
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.extract import parse_annotation, resize_bbox_1
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco


parser = argparse.ArgumentParser()

parser.add_argument('-eval_path', type=str)
parser.add_argument('-pre_path', type=str)

args = parser.parse_args()

root_path = args.eval_path
label_path = args.pre_path

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_cocohazed_6.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on

#root_path = "track2.1_test_sample"
#label_path = "results"
image_dir = os.path.join(root_path)
annotations_dir = os.path.join(label_path)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorbike', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_PADDING = True

config = InferenceConfig()
config.display()
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

def convert_to_xml(image_dir, annotations_dir):
    APs = []
    image_names = next(os.walk(image_dir))[2]
    image_names.sort()
    for image_name in image_names:
        image = skimage.io.imread(os.path.join(image_dir, image_name))
        image, window, scale, padding = utils.resize_image_test(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)
        if image_name.find('AOD') != -1:
            image_name = image_name.replace('_AOD-Net', '')
        if image_name.find('dehaze') != -1:
            image_name = image_name.replace('_dehazed', '')

        results = model.detect([image], verbose=1)
        r = results[0]
        rois = r["rois"]
        rois = resize_bbox_1(rois, scale, padding)
        class_ids = r["class_ids"]
        scores = r["scores"]

        annotation = ET.Element('annotation')
        filename = ET.SubElement(annotation, 'filename')
        filename.text = image_name[:-4]
        for j in range(len(rois)):


            if class_names[class_ids[j]] == 'person' or class_names[class_ids[j]] == 'bus' or class_names[class_ids[j]] == 'bicycle' \
                    or class_names[class_ids[j]] == 'car' or class_names[class_ids[j]] == 'motorbike':
                Object = ET.SubElement(annotation, 'object')
                name = ET.SubElement(Object, 'name')
                name.text = class_names[class_ids[j]]

                difficult = ET.SubElement(Object, 'difficult')
                difficult.text = str(1 - scores[j])

                bndbox = ET.SubElement(Object, 'bndbox')
                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = str(rois[j][1])
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = str(rois[j][0])
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(rois[j][3])
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(rois[j][2])
                
        tree = ET.ElementTree(annotation)
        tree.write(annotations_dir + "/" + image_name[:-4] + ".xml")

cur_mAP2 = convert_to_xml(image_dir, annotations_dir)