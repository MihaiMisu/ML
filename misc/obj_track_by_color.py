# -*- coding: utf-8 -*-

import cv2
import numpy as np

rgb_color = np.uint8([[[0, 165, 255]]])  # ORANGE
#rgb_color = np.uint8([[[255, 255, 255]]])
hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_BGR2HSV)

tollerance = 20

lower_hsv_color = hsv_color[0][0][0] - tollerance, 100, 100
lower_hsv_color = np.array(hsv_color)

upper_hsv_color = hsv_color[0][0][0] + tollerance, 255, 255
upper_hsv_color = np.array(hsv_color)

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([64, 98, 142])
    upper_blue = np.array([0, 110, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_hsv_color, upper_hsv_color)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
