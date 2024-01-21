from cvzone.FaceDetectionModule import FaceDetector
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
detector = FaceDetector()
################################
offsetpercentageW=10
offsetpercentageH=20

################################

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)

    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        for bbox in bboxs:

            x,y,w,h=bbox["bbox"]
            print(x,y,w,h)

            offsetW=(offsetpercentageW/100)*w
            offsetH=(offsetpercentageH/100)*h

            x=int(x-offsetW)
            w=int(w+offsetW*2)
            y=int(y-offsetH*2.5)
            h=int(h+offsetH*3)
            cv2.rectangle(img,(x,y,w,h),(255,255,34))




    cv2.imshow("Image", img)
    cv2.waitKey(1)
