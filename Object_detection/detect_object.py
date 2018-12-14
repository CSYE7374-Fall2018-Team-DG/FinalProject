#!/usr/bin/env python


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from flask import Flask, render_template, flash, request
#from gtts import gTTS
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
#from google.cloud import texttospeech
import array
# from tempfile import TemporaryFile
# import playsound
# from pygame import mixer

from win32com.client import constants, Dispatch

import subprocess
#import pyttsx

import cv2


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# ## Object detection imports
# Here are the imports from the object detection module.


from utils import label_map_util
from time import sleep
from utils import visualization_utils as vis_util

def real_time():# What model to download.
  cap = cv2.VideoCapture(0)
  MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
  MODEL_FILE = MODEL_NAME + '.tar.gz'
  DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
#APP_PORT = 5000
# Path to frozen detection graph. This is the actual model that is used for the object detection.
  PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
  PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# Download Model
  opener = urllib.request.URLopener()
  opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
  tar_file = tarfile.open(MODEL_FILE)
  for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
      tar_file.extract(file, os.getcwd())

# Load a (frozen) Tensorflow model into memory.
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
  category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

  print('stage 1')
  with detection_graph.as_default():
    print('stage 2')
    with tf.Session(graph=detection_graph) as sess:
      print('stage 3')
      while True:
        print('stage 4')
        
        ret, image_np = cap.read()
        print(ret)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
# Actual detection 
        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
# Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,line_thickness=8)
        print('stage 5')
        #print([category_index.get(i) for i in classes[0]])
        cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
        final_score = np.squeeze(scores) 
        count = 0
        for i in range(100):
          if scores is None or final_score[i] > 0.5:  
            count = count + 1
        print('count',count)
        for i in classes[0]:
          print(category_index[i]['name'])
          detected_label = category_index[i]['name']
          printcount = printcount +1
          if(printcount == count):
            break

        if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break
  return detected_label

if __name__ == "__main__":
  real_time()