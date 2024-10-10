#multiple pred test
from ultralytics import YOLO
import tensorflow as tf
import imutils, cv2
import os, time, keyboard
import argparse, copy
from pathlib import Path
from source.numplate_detection import numberplate_detection, bbox_and_labels, save_pred, numplate_model




def numberplate_recognition(source):

    start = time.time()
    model = numplate_model
    save_dir = "runs"
    yolo_model = YOLO("models/yv8.pt")
    os.makedirs(save_dir, exist_ok=True)
    
    if os.path.isdir(source):
        # Iterate over all files
        for filename in os.listdir(source):
            image_path = os.path.join(source, filename)
            if os.path.isfile(image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):

                bounding_boxes, labels, img = numberplate_detection(image_path,yolo_model,model)
                fname = os.path.splitext(filename)[0]
                input_image = copy.copy(img)
                savepath, img = bbox_and_labels(img,bounding_boxes,labels,save_dir,filename)

                resized = imutils.resize(img, width=640)
                ip_resized = imutils.resize(input_image, width=640)
                cv2.imshow("input image", ip_resized)
                cv2.imshow("output figure",resized)
                cv2.waitKey(0)

                textfile = save_pred(fname,bounding_boxes,labels,save_dir)
                print(f"predicted to {savepath} \npredictions saved to {textfile}")
                
    elif os.path.isfile(source):
        bounding_boxes, labels, img = numberplate_detection(source,yolo_model,model)
        fname = os.path.basename(source)
        filename = os.path.splitext(fname)[0]
        savepath, img = bbox_and_labels(img,bounding_boxes,labels,save_dir,fname)
        resized = imutils.resize(img, width=640)
        cv2.imshow("output figure",resized)
        cv2.waitKey(0)
        textfile = save_pred(filename,bounding_boxes,labels,save_dir)
        print(f"predicted to {savepath} \npredictions saved to {textfile}")

    else:
        print(f"Source path '{source}' is neither a valid directory nor a file.")

    cv2.destroyAllWindows()
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")


def main():

    
    parser = argparse.ArgumentParser(description="Predicts the characters in the arabic numberplate")
    parser.add_argument('Image_files', type=str, help="Path to image file or folder path"
                        )
    args = parser.parse_args()
    print(f"Detecting plates from :{args.Image_files}")
    numberplate_recognition(args.Image_files)



if __name__ == "__main__":
    main()

    