# -*- coding: utf-8 -*-

#%%   WEB CAM FACE DETECTION USING openCV  ALGORITHM
#%%

#from cv2 import (CascadeClassifier, rectangle, imshow, waitKey, VideoCapture,
#                 namedWindow, destroyWindow)
#
#classifier = CascadeClassifier("trained_nn_face_detection.xml")
#
#namedWindow("preview")
#vc = VideoCapture(0)
#
#if vc.isOpened(): # try to get the first frame
#    rval, frame = vc.read()
#else:
#    rval = False
#
#while rval:
#    imshow("preview", frame)
#    rval, frame = vc.read()
#    
#    bboxes = classifier.detectMultiScale(frame, 1.05, 3)
#    for box in bboxes:
##	print(box)
#        x, y, width, height = box
#        x2, y2 = x + width, y + height
#    # draw a rectangle over the pixels00
#        rectangle(frame, (x, y), (x2, y2), (0,0,255), 1)
#    
#    key = waitKey(20)
#    if key == 27: # exit on ESC
#        break
#
#vc.release()
#destroyWindow("preview")


#%%   WEB CAM FACE DETECTION USING DEEPLEARNING
#%%

#from mtcnn.mtcnn import MTCNN
from mtcnn_copy import MTCNN
from cv2 import (CascadeClassifier, rectangle, imshow, waitKey, VideoCapture,
                 namedWindow, destroyWindow, circle, destroyAllWindows)

detector = MTCNN()

namedWindow("preview")
vc = VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    faces = detector.detect_faces(frame)
    
    for face in faces:
   
        x, y, width, height = face['box']
        # create the shape
        rectangle(frame, (x, y), (x+width, y+height), (0,0,255), 1)
    
    imshow("preview", frame)
    rval, frame = vc.read()
    key = waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
destroyWindow("preview")



#cap = VideoCapture(0)
#while True: 
#    #Capture frame-by-frame
#    __, frame = cap.read()
#    
#    #Use MTCNN to detect faces
#    result = detector.detect_faces(frame)
#    if result != []:
#        for person in result:
#            bounding_box = person['box']
#            keypoints = person['keypoints']
#    
#            rectangle(frame,
#                          (bounding_box[0], bounding_box[1]),
#                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
#                          (0,155,255),
#                          2)
#    
#            circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
#            circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
#            circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
#            circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
#            circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)
#    #display resulting frame
#    imshow('frame',frame)
#    if waitKey(1) &0xFF == ord('q'):
#        break#When everything's done, release capture
#cap.release()
#destroyAllWindows()



