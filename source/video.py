import os
import cv2
from ultralytics import YOLO
import arabic_reshaper
from bidi.algorithm import get_display
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from source.ocr import build_model
from source.utils import decode_label, label_to_ar, label_to_en
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import ImageFont, ImageDraw, Image

yolo_model = YOLO("/home/arvind/Python/DL/ENPD/models/yv8.pt")
video_path = '/home/arvind/Python/DL/ENPD/data/input.mp4'
output_video_path = 'data/output_video.mp4'

numplate_model = build_model(False)
numplate_model.load_weights("models/best_weight.h5")


cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model.predict(frame, conf = 0.1)
    font = ImageFont.truetype(font= "arial.ttf", size=20)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes: 
            r = box.xyxy[0].astype(int)
            #cv2.rectangle(frame, r[:2], r[2:], (0, 255, 0), 2)  # Draw boxes on the image
            crop_image = frame[r[1]:r[3],r[0]:r[2]]

            img_grey = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        
            #process the img
            img_pred = img_grey.astype(np.float32)
            img_pred = cv2.resize(img_pred, (128, 64))
            img_pred = (img_pred / 255.0)
            img_pred = img_pred.T
            img_pred = np.expand_dims(img_pred, axis=-1)
            img_pred = np.expand_dims(img_pred, axis=0)


            outp = numplate_model.predict(img_pred, verbose=1)
            pred_text = decode_label(outp)
            pred_text_ar = label_to_ar(pred_text)
            print(pred_text)
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            draw.rectangle([r[0], r[1], r[2], r[3]], outline=(0,255,0), width=2)

            #arabic text label
            reshaped_ar_label = arabic_reshaper.reshape(pred_text_ar)
            bidi_ar_text = get_display(reshaped_ar_label)
            text_x1, text_y1, text_x2, text_y2 = draw.textbbox((r[0],r[1]),bidi_ar_text, font=font)         #get the text size

            height = (text_y2 - text_y1) + 5
            width = (text_x2 - text_x1) + 5

            draw.rectangle([(r[0], r[1]-height), (r[0]+width, r[1])], (0,255,0), width=2)
            draw.text((r[0]+3 , r[1]-height), bidi_ar_text, font=font, fill=(0, 0, 0))

            #english label
            reshaped_en_label = label_to_en(pred_text_ar)
            bidi_en_text = get_display(reshaped_en_label)
            text1_x1, text1_y1, text1_x2, text1_y2 = draw.textbbox((r[0],r[1]),bidi_en_text, font=font)

            height1 = (text1_y2 - text1_y1) + 5
            width1 = (text1_x2 - text1_x1) + 5

            draw.rectangle([(r[0], r[3]), (r[0]+width1, r[3]+height1)], (0,255,0), width=2)
            draw.text((r[0]+3 , r[3]), bidi_en_text, font=font, fill=(0, 0, 0))

            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


    cv2.imshow('Frame',frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
 
        

    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
