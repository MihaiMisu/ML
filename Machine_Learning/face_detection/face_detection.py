# -*- coding: utf-8 -*-

#%%   FACE DETECTION USING openCV - ALGORITHMS
#%%
from cv2 import CascadeClassifier, imread, rectangle, imshow, destroyAllWindows, waitKey
from os import getcwd

print(getcwd())

# load the pre-trained model
classifier = CascadeClassifier("trained_nn_face_detection.xml")
	
# load the photograph
pixels = imread('test2.jpg')

# perform face detection
bboxes = classifier.detectMultiScale(pixels, 1.05, 3)
# print bounding box for each detected face
for box in bboxes:
#	print(box)
    x, y, width, height = box
    x2, y2 = x + width, y + height
    # draw a rectangle over the pixels00
    rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
	
# show the image
imshow('face detection', pixels)
# keep the window open until we press a key
waitKey(0)
# close the window
destroyAllWindows()


#%%   FACE DETECTION USING DEEP LEARNING
#%%
	
# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.pyplot import Rectangle
from mtcnn.mtcnn import MTCNN
# load image from file
filename = 'test2.jpg'
pixels = pyplot.imread(filename)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
for face in faces:

    pyplot.imshow(pixels)    
    ax = pyplot.gca()
    # get coordinates
    x, y, width, height = face['box']
    # create the shape
    rect = Rectangle((x, y), width, height, fill=False, color='red')
    ax.add_patch(rect)

pyplot.show()













