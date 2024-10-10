from ultralytics import YOLO
import tensorflow as tf
import cv2
import itertools, os, time
import numpy as np
import argparse
import pandas as pd
import operator
import os
from source.ocr import build_model
from source.utils import decode_label, label_to_ar, label_to_en


numplate_model = build_model(False)
numplate_model.load_weights("models/best_weight.h5")



#multi pred test

import os
import cv2
from ultralytics import YOLO
import arabic_reshaper
from bidi.algorithm import get_display
import numpy as np
from PIL import ImageFont, ImageDraw, Image



def numberplate_detection(image_path, yolo_model, model):
    img = cv2.imread(image_path)
    results = yolo_model.predict(image_path)
    bounding_boxes = []
    
    # if detections were made
    if results[0].boxes is not None:
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])  # Get bounding box coordinates
            bounding_boxes.append((x1, y1, x2, y2))
    
    labels = []

    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        crop_img = img[y1:y2, x1:x2]  # Crop the image
        #converting to bw
        img_grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        #process the img
        img_pred = img_grey.astype(np.float32)
        img_pred = cv2.resize(img_pred, (128, 64))
        img_pred = (img_pred / 255.0)
        img_pred = img_pred.T
        img_pred = np.expand_dims(img_pred, axis=-1)
        img_pred = np.expand_dims(img_pred, axis=0)

        #predict on img
        net_out_value = model.predict(img_pred, verbose=1)
        pred_text = decode_label(net_out_value)
        pred_text_ar = label_to_ar(pred_text)

        labels.append(pred_text_ar)

    print(labels)
    return bounding_boxes, labels, img



        

def bbox_and_labels(image, bbox, labels, save_dir, filename):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype(font= "arial.ttf", size=20)

    for box, label in zip(bbox,labels):
        x1, y1, x2, y2 = box

        #bbox
        draw.rectangle([x1, y1, x2, y2], outline=(0,255,0), width=2)

        #arabic text label
        reshaped_ar_label = arabic_reshaper.reshape(label)
        bidi_ar_text = get_display(reshaped_ar_label)
        text_x1, text_y1, text_x2, text_y2 = draw.textbbox((x1,y1),bidi_ar_text, font=font)         #get the text size

        height = (text_y2 - text_y1) + 5
        width = (text_x2 - text_x1) + 5

        draw.rectangle([(x1, y1-height), (x1+width, y1)], (0,255,0), width=2)
        draw.text((x1+3 , y1-height), bidi_ar_text, font=font, fill=(0, 0, 0))

        #english label
        reshaped_en_label = label_to_en(label)
        bidi_en_text = get_display(reshaped_en_label)
        text1_x1, text1_y1, text1_x2, text1_y2 = draw.textbbox((x1,y1),bidi_en_text, font=font)

        height1 = (text1_y2 - text1_y1) + 5
        width1 = (text1_x2 - text1_x1) + 5

        draw.rectangle([(x1, y2), (x1+width1, y2+height1)], (0,255,0), width=2)
        draw.text((x1+3 , y2), bidi_en_text, font=font, fill=(0, 0, 0))

    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    #os.makedirs(save_dir, exist_ok=True)

    # Save the image with bounding boxes and labels
    save_path = os.path.join(save_dir, os.path.basename(filename))
    cv2.imwrite(save_path, img)

    return save_path, img


def save_pred(image_name, bbox, labels, save_dir):

    txt_folder = os.path.join(save_dir, "labels")
    os.makedirs(txt_folder, exist_ok=True)
    txt_file = os.path.join(txt_folder, f"{image_name}.txt")

    with open (txt_file, 'w', encoding='utf-8') as txt:
        for box, label in zip(bbox, labels):
            x1, y1, x2, y2 = box
            en_label = label_to_en(label)
            txt.write(f"Label : {label}, Eng_label : {en_label}, Coordinates : {x1} {y1} {x2} {y2}")
    return txt_file

