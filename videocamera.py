import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow("grey", gray)

    haar_cascade = cv.CascadeClassifier('haar_face.xml')

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
    cv.imshow("frame", frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()