import cv2
import numpy as np
import random

demo_file = "mmdetection3d/data/own/demo.mp4"

cap = cv2.VideoCapture(demo_file)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    color = list(np.random.random(size=3) * 256)
    frame = cv2.rectangle(frame, (random.randrange(300,900), random.randrange(200,500)), (random.randrange(100,200), random.randrange(100,200)), color, 4)
    cv2.imshow("demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
