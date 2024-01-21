import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import mediapipe as mp
from time import time
################################
classTD =0 # 0 is fake 1 is real
confidence = 0.8
save= True
blurTreshold = 35
outputFolderPath='Dataset/DataCollect'

debug = False
camwidth, camheight = 640, 480
floatingpoint = 6
offsetpercentageW = 10
offsetpercentageH = 20
################################

cap = cv2.VideoCapture(0)
cap.set(3, camwidth)
cap.set(4, camheight)
detector = FaceDetector()

while True:
    success, img = cap.read()
    imgOut=img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur= [] #true fals value indicating if the faces are blur or not
    listInfo=[] # the normalized value and the class names for the label text file


    if bboxs:
        # bboxInfo - "id", "bbox", "score", "center"
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]

            # Check the score
            if score > confidence:
                # Adding an offset to the face
                offsetW = (offsetpercentageW / 100) * w
                offsetH = (offsetpercentageH / 100) * h
                x = int(x - offsetW)
                w = int(w + offsetW * 2)
                y = int(y - offsetH * 2.5)
                h = int(h + offsetH * 3)

                # To avoid values below 0
                x = max(0, x)
                y = max(0, y)
                h = max(0, h)
                w = max(0, w)

                # Find blurriness
                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())

                if blurValue > blurTreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                # Normalization Value
                ih, iw, _ = img.shape
                xc = x + w / 2
                yc = y + h / 2
                xcn = round(xc / iw, floatingpoint)
                ycn = round(yc / ih, floatingpoint)
                wn= round(w / iw, floatingpoint)
                hn = round(h / ih, floatingpoint)
                #print(xcn, ycn,wn,hn)
                # To avoid values above 1
                xcn = min(1, xcn)
                ycn = min(1, ycn)
                hn = min(1, hn)
                wn = min(1, wn)

                listInfo.append(f"{classTD} {xcn} {ycn} {wn} {hn}\n")

                # Drawing
                cv2.rectangle(imgOut, (x, y), (x + w, y + h), (255, 255, 34))
                cvzone.putTextRect(imgOut, f"Score:{int(score*100)}blur:{blurValue}", (x, y - 20),scale=1,thickness=1)

                if debug :
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 34))
                    cvzone.putTextRect(img, f"Score:{int(score * 100)}blur:{blurValue}", (x, y - 20), scale=1,
                                       thickness=1)

                cv2.imshow("Image", imgOut)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        #---------- to Save ----------#

        if save:
            if all(listBlur) and listBlur!= []:

                #----------- save image-----------#
                timeNow=time()
                timeNow = str(timeNow).split('.')
                timeNow= timeNow[0]+ timeNow[1]

                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg",img)

                #---------- save label Text file ---------------#

                f= open("test.txt",'a')
                f.write()
                f.close()


cap.release()
cv2.destroyAllWindows()
