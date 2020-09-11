from collections import deque
from imutils.video import VideoStream
import argparse
import numpy as np
import cv2 as cv
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained models")
ap.add_argument("-c", "--confidence", type=float, 
                default=0.5,
                help="min probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading model...")
net= cv.dnn.readNetFromCaffe(args["prototxt"], args["model"])
vs = VideoStream(src=0).start()
time.sleep(2.0)
pts = deque(maxlen=64)

vid = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vid.read()
    resize = imutils.resize(frame, height=500)
    (h,w) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(frame,(300,300)), 1.0,
                                (300,300), (104.,177.,123.))
    net.setInput(blob)
    detections = net.forward()
    center = None
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > args["confidence"]:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence*100)
            y = startY - 10 if startY-10>10 else startY+10
            center = (int((endX-startX)/2), int((endY-startY)/2))
            cv.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)
    pts.appendleft(center)
    
    for i in range(1, len(pts)):
        if pts[i-1] is None or pts[i] is None:
            continue
        cv.line(frame, pts[i-1], pts[i], (0,0,255), 2)
    
    cv.imshow('stream', frame)
    k = cv.waitKey(0)
    if k == ord('q'):
        cv.destroyAllWindows()