# -*- coding: utf-8 -*-



from cv2 import (CascadeClassifier, rectangle, imshow, waitKey, VideoCapture,
                 namedWindow, destroyWindow)

classifier = CascadeClassifier("trained_nn_face_detection.xml")

namedWindow("preview")
vc = VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    imshow("preview", frame)
    rval, frame = vc.read()
    
    bboxes = classifier.detectMultiScale(frame, 1.05, 3)
    for box in bboxes:
#	print(box)
        x, y, width, height = box
        x2, y2 = x + width, y + height
    # draw a rectangle over the pixels00
        rectangle(frame, (x, y), (x2, y2), (0,0,255), 1)
    
    key = waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
destroyWindow("preview")