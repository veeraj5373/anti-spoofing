from ultralytics import YOLO
import cv2
import numpy as np
import cvzone
import math
import time

model=YOLO('../../models/yolov8n.pt')

classnames=["person", "bicycle", "car" , "motorbike","aeroplane","bUs","train", "truck", "boat",
            "traffic Light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag" ,"tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove","skateboard","surfboard" , "tennis racket", "bottle", "wine glass","cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple","sandwich""orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
            "dining table", "toilet", "tv monitor", "Laptop", "mouse","remote","keyboard","cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book","clock","vase","scissors",
            "teddy bear", "hair drier", "toothbrush"]
prev_frame_time=0
new_frame_time=0
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

while True:
    new_frame_time=time.time()
    success , img = cap.read(0)
    if not success:
        break

    results = model(img,device="mps",verbose= False)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
    confs= np.array(result.boxes.conf.cpu(), dtype="float")
    classes= np.array(result.boxes.cls.cpu(),dtype="int")
    for cls,bbox, conf in zip(classes,bboxes,confs):
        (x,y,x2,y2)=bbox
        currentClass = classnames[cls]
        conf=math.ceil((conf*100))/100

        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 255), 2)
        cv2.putText(img,f"{classnames[int(cls)]} {conf}",(x,y-5),cv2.FONT_HERSHEY_PLAIN,1,(0,0,225),2)

    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time=new_frame_time
    print(fps)

    cv2.imshow('Image',img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()