import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import mediapipe as mp


################################
offsetpercentageW=10
offsetpercentageH=20
confidence=0.8
camwidth , camheight = 640,480
floatingpoint=6
################################

cap = cv2.VideoCapture(0)
cap.set(3,camwidth)
cap.set(4,camheight)
detector = FaceDetector()


while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img,draw=False)

    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        for bbox in bboxs:

            x,y,w,h=bbox["bbox"]
            score=bbox["score"][0]
            print(x,y,w,h)
            # ------------- Check the score -----------#
            if score >confidence:




                # ------------- Adding an offset to the face -----------#

                offsetW=(offsetpercentageW/100)*w
                offsetH=(offsetpercentageH/100)*h

                x=int(x-offsetW)
                w=int(w+offsetW*2)
                y=int(y-offsetH*2.5)
                h=int(h+offsetH*3)


                #------------- To avoid value below 0 -----------#
                if x<0:x=0
                if y<0:y=0
                if h < 0: h = 0
                if w < 0: w = 0


            #---------find blurriness-----------#
                imgFace= img[y:y+h,x:x+w]
                cv2.imshow("Face", imgFace)
                blueValue=int(cv2.Laplacian(imgFace, cv2.CV_64F).var())

            # ----------- Normalization Value ------------#
                ih , iw ,_=img.shape
                xc = x+w/2
                yc = y=y+h/2
                xcn= round(xc/iw,floatingpoint)
                ycn= round(yc/ih,floatingpoint)
                print(xcn ,ycn)




            #-----------Drawing------------#
                cv2.rectangle(img,(x,y), (x+ y, w+ h), (255, 255, 34),2)
                cvzone.putTextRect(img,f"blur:{blueValue}",(x,y-20))

    cv2.imshow("Image", img)
    cv2.waitKey(1)