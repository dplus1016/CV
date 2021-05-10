#youtube: https://www.youtube.com/watch?v=p5Z_GGRCI5s
#website: https://www.computervision.zone/courses/finger-counter/

import cv2
import time
import os
import HandTrackingModule as htm
 
wCam, hCam = 640, 480
 
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
 
overlayList = []
pTime = 0
 
detector = htm.handDetector(detectionCon=0.75)
 
tipIds = [4, 8, 12, 16, 20]
 
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    print(lmList)
 
    if len(lmList) != 0:
        fingers = []
        print(lmList)
        
        direction=0
        if lmList[tipIds[0]][1] < lmList[tipIds[4]][1]: direction=1
 
        print(f"direction = {direction}")

        # Thumb
        if lmList[tipIds[0] - direction][1] > lmList[tipIds[0] - (1-direction)][1]:
            fingers.append(1)
        else:
            fingers.append(0)
 
        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
 
        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)

    # 키보드 조작
    if cv2.waitKey(1) & 0xFF == ord('q'): break  # q: 종료

 
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
 
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
 
    cv2.imshow("Image", img)

cv2.destroyAllWindows()
