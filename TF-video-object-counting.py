#!/usr/bin/env python
# coding: utf-8
"""
Object Detection (On Video) From TF2 Saved Model
=====================================
"""

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import cv2
import argparse
import math

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Folder that the Saved Model is Located In',
                    default='exported-models/my_model')
parser.add_argument('--labels', help='Where the Labelmap is Located',
                    default='annotations/label_map.pbtxt')
parser.add_argument('--video', help='Name of the video to perform detection on. To run detection on multiple images, use --imagedir',
                    default='image-test/cam1.mp4')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
                    
args = parser.parse_args()
# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PROVIDE PATH TO IMAGE DIRECTORY
VIDEO_PATHS = args.video


# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = args.model

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = args.labels

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(args.threshold)

# Load the model
# ~~~~~~~~~~~~~~
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
from tkinter import *
import numpy as np
from PIL import Image,ImageTk
import math
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

#tham so
max_distance = 50


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


def get_object(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)


    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(frame)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # SET MIN SCORE THRESH TO MINIMUM THRESHOLD FOR DETECTIONS

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    scores = detections['detection_scores']
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    box = []
    for i in range(len(scores)):
        if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):
            # increase count

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            boxx=[xmin,ymin,xmax-xmin, ymax-ymin]
            box.append(boxx)
    return box

def is_old(center_Xd, center_Yd, boxes):
    for box_tracker in boxes:
        (xt, yt, wt, ht) = [int(c) for c in box_tracker]
        center_Xt, center_Yt = int((xt + (xt + wt)) / 2.0), int((yt + (yt + ht)) / 2.0)
        distance = math.sqrt((center_Xt - center_Xd) ** 2 + (center_Yt - center_Yd) ** 2)

        if distance < max_distance:
            return True
    return False


def get_box_info(box):
    (x, y, w, h) = [int(v) for v in box]
    center_X = int((x + (x + w)) / 2.0)
    center_Y = int((y + (y + h)) / 2.0)
    return x, y, w, h, center_X, center_Y


print('Running inference for {}... '.format(VIDEO_PATHS), end='')



# khoi tao tham so
imH = 500
imW = 700
frame_count = 0
car_number = 0
obj_cnt = 0
curr_trackers = []


'''window=Tk()
window.title("Traffic App")




vd_w = video.get(cv2.CAP_PROP_FRAME_WIDTH)*3//4
vd_H = video.get(cv2.CAP_PROP_FRAME_HEIGHT)*3//4

canvas = Canvas(window, width = imW,heigh= imH)
canvas.pack()'''
video = cv2.VideoCapture(VIDEO_PATHS)

while (video.isOpened()):
    # global canvas, img,curr_trackers,car_number,frame_count,obj_cnt

    boxes = []
    ret, frame = video.read()
    frame = cv2.resize(frame, (imW, imH))

    x_point_1 = int(imW/2)
    y_point_1 = imH
    x_point_2 = imW
    y_point_2 = int(imH/2)
    distance_1_2 = math.sqrt((x_point_2-x_point_1)**2 + (y_point_2-y_point_1)**2)

    laser_line= imH-100
    laser_line_color = (0, 0, 255)
    old_trackers = curr_trackers
    curr_trackers = []

    #duyệt qua các tracker cũ
    for car in old_trackers:
        tracker=car['tracker']
        (_, box) = tracker.update(frame)
        boxes.append(box)

        new_obj = dict()
        new_obj['tracker_id'] = car['tracker_id']
        new_obj['tracker'] = tracker

        #tính toán tâm đối tượng
        x, y, w, h, center_X, center_Y = get_box_info(box)

        # Ve hinh chu nhat quanh doi tuong
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Ve hinh tron tai tam doi tuong
        cv2.circle(frame, (center_X, center_Y), 4, (0, 255, 0), -1)

        # So sanh tam doi tuong voi duong laser line
        distance1= center_X*5 + center_Y*7 - 5250
        distance2 = center_X*13 +center_Y*11 - 3850

        if distance1 <100 and distance1>0:
            # Neu vuot qua thi khong track nua ma dem xe
            laser_line_color = (0, 255, 255)
            car_number += 1

        else:
            # Con khong thi track tiep
            curr_trackers.append(new_obj)

    # Thuc hien object detection moi 5 frame
    if frame_count % 5 == 0:
        # Detect doi tuong
        boxes_d = get_object(frame)

        for box in boxes_d:
            old_obj = False

            xd, yd, wd, hd, center_Xd, center_Yd = get_box_info(box)
            xline1 = int((5250-7*center_Yd)/5)
            xline2 = int((3850-11*center_Yd)/13)
            xline3 = int(10 + 4 * center_Yd)
            if center_Xd <= xline1 :

                # Duyet qua cac box, neu sai lech giua doi tuong detect voi doi tuong da track ko qua max_distance thi coi nhu 1 doi tuong
                if not is_old(center_Xd, center_Yd, boxes):
                    cv2.rectangle(frame, (xd, yd), ((xd + wd), (yd + hd)), (0, 255, 255), 2)
                    # Tao doi tuong tracker moi

                    tracker = cv2.TrackerMOSSE_create()

                    obj_cnt += 1
                    new_obj = dict()
                    tracker.init(frame, tuple(box))

                    new_obj['tracker_id'] = obj_cnt
                    new_obj['tracker'] = tracker

                    curr_trackers.append(new_obj)

                # Tang frame
    frame_count += 1

    # Hien thi so xe

    cv2.putText(frame, "Car number: " + str(car_number), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(frame, "Press Esc to quit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Draw laser line
    cv2.line(frame, (x_point_1,y_point_1), (x_point_2,y_point_2 ), laser_line_color, 2)
    cv2.line(frame, (220, 90), (165, 155), laser_line_color, 2)
    cv2.line(frame, (450, 110), (700, 165), laser_line_color, 2)
    cv2.line(frame, (0, 240), (320, 500), laser_line_color, 2)

    #cv2.putText(frame, "Laser line", (10, laser_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, laser_line_color, 2)
    '''frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(image=Image.fromarray(frame))
    canvas.create_image(0, 0, image=img, anchor=NW)
    window.after(5, update_frame)'''
    # Frame
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
#update_frame()
#window.mainloop()
video.release()
cv2.destroyAllWindows()
print("Done")
