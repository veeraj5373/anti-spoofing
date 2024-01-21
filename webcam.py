from ultralytics import YOLO
import cv2
import cvzone
import math


cap = cv2.VideoCapture(0)
model=YOLO('../yolo-weights/yolov8n.pt')

classnames=["person", "bicycle", "car" , "motorbike","aeroplane","bUs","train", "truck", "boat",
            "traffic Light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag" "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove","skateboard","surfboard" , "tennis racket", "bottle", "wine glass","cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple","sandwich""orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
            "diningtable", "toilet", "tvmonitor", "Laptop", "mouse","remote","keyboard","cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book","clock","vase","scissors",
            "teddy bear", "hair drier", "toothbrush"]

cap.set(3,1280)
cap.set(4,720)

while True:
    success, img = cap.read()
    results = model(img,stream=True,device="mps")
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 =box.xyxy[0]
            x1, y1, x2, y2 =int(x1),int(y1),int(x2),int(y2)
            w ,h = x2-x1,y2-y1

            cvzone.cornerRect(img,(x1,y1,w,h))

            conf=math.ceil((box.conf[0]*100))/100
            cls=box.cls[0]

            cvzone.putTextRect(img, text=f'{classnames[int(cls)]}{conf}',pos=(max(0,x1),max(40,y1)) ,scale=0.7 , thickness=1)


    cv2.imshow('Image',img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
