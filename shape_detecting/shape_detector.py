# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:29:16 2020

@author: Mihai

Script which is supposed to extract the center of different shapes (triangles,
    squares, rectangles, hexathings, etc) and eventually to mark parimeter.

The program uses openCV as main image processing tool and another obscure
    library - imutils. There are 2 possibilities of using a static or dynamic
    thresholding algorithm for image binarization. It depends on the type of
    the image in the way of how complex it is (only geometrical shapes or more
    complex things) to use one of them.

Input parameters to set:
    -> image path: quite obvious what it is and what it does. Name: IMG_PATH;
    -> plotting graphs: if there is wanted to visualize graphic intermediate
    results. Name: ENABLE_PLOTTING
    -> using adaptive threshold algorithm ot not. Name: ADAPTIVE_ALGO_USE

"""

# %%     IMPORTS
# %%
import matplotlib.pyplot as plt

from os import chdir, getcwd
from os.path import (join, dirname, abspath)
from cv2 import (imread as cv_imread, resize as cv_resize,
                 cvtColor as cv_cvtColor, GaussianBlur as cv_GaussBlur,
                 bilateralFilter as cv_bilateralFilter,
                 threshold as cv_tHold, adaptiveThreshold as cv_adaptiveThold,
                 findContours as cv_findContours, circle as cv_circle,
                 drawContours as cv_drawContours, putText as cv_putText,
                 moments as cv_moments, arcLength as cv_arcLength,
                 approxPolyDP as cv_approxPolyDP,
                 boundingRect as cv_boundingRect,
                 COLOR_BGR2RGB, COLOR_RGB2GRAY,
                 THRESH_BINARY, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_OTSU,
                 RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, FONT_HERSHEY_SIMPLEX, RETR_TREE,
                 INTER_CUBIC, INTER_LINEAR,)

# %%     CONSTANTS
# %%
chdir(dirname(abspath(__file__)))
print(F"Current working directory: {getcwd()}")

#IMG_PATH = join(getcwd(), "..", "mak_rcnn_pytorch", "Receipts", "IMG_1167.JPG")
IMG_PATH = join(getcwd(), "..", "shape_detecting", "shapes_and_colors.jpg")

ENABLE_PLOTTING = True
ADAPTIVE_ALGO_USE = False

# %%     FUNCTIONS
# %%

# %%     CLASSES
# %%


class ShapeDetector:

    def __init__(self):
        pass

    def detect(self, c):
        """
        Method which will return the type of the shape it detects
        :param c: contour used to detect shape
        """
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv_arcLength(c, True)
        approx = cv_approxPolyDP(c, 0.04 * peri, True)

        # he shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
            # if the shape has 4 vertices, it is either a square or
            # a rectangle

        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv_boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
            # if the shape is a pentagon, it will have 5 vertices

        elif len(approx) == 5:
            shape = "pentagon"
            # otherwise, we assume the shape is a circle

        else:
            shape = "circle"

        # return the name of the shape
        return shape

# %%
# %%
# READ IMAGE AS BGR
image = cv_imread(IMG_PATH)

# Image resizing with different methodes: CUBIC/LINERA interpolation
# image = cv_resize(image, None, fx=1/2, fy=1/2, interpolation=INTER_CUBIC)
image = cv_resize(image, None, fx=1/2, fy=1/2, interpolation=INTER_LINEAR)

# CONVERT IMAGE FROM BGR (OPENCV DEFAULT READING FORMAT) TO RGB
image = cv_cvtColor(image, COLOR_BGR2RGB)

# CONVERT IMAGE TO GRAY AND BLUR IT USING GAUSS LOW PASS FILTER
gray_img = cv_cvtColor(image, COLOR_RGB2GRAY)
blur_img = cv_GaussBlur(gray_img, (5, 5), 0)

# FILTERING
# Apply threshold on image
ret_val_1, static_img = cv_tHold(blur_img, 127, 255, THRESH_BINARY)
# 5th parameter has to be an odd number
adaptive_img = cv_adaptiveThold(blur_img, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                                THRESH_BINARY, 101, 5)
# OTSU filtering
ret_val_2, otsu_img = cv_tHold(blur_img, 0, 255, THRESH_BINARY+THRESH_OTSU)
# Bilateral filtering
bilateral_img = cv_bilateralFilter(adaptive_img, 9, 75, 75)

# Contour extractig
static_contour, _ = cv_findContours(static_img.copy(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
adaptive_contour, _ = cv_findContours(adaptive_img.copy(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
otsu_contour, _ = cv_findContours(otsu_img.copy(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
bilateral_contour, _ = cv_findContours(bilateral_img.copy(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

static_img_contour = cv_drawContours(static_img, static_contour, -1, (0,255,0), 3)
adaptive_img_contour = cv_drawContours(adaptive_img, adaptive_contour, -1, (0,255,0), 3)
otsu_img_contour = cv_drawContours(otsu_img, adaptive_contour, -1, (0,255,0), 3)
bilinear_img_contour = cv_drawContours(bilateral_img, adaptive_contour, -1, (0,255,0), 3)

shape_detect = ShapeDetector()
# loop over the contours
for c in static_contour:
    # compute the center of the contour
    M = cv_moments(c)
    cX = int(M["m10"] / M["m00"]) if M["m00"] else 0
    cY = int(M["m01"] / M["m00"]) if M["m00"] else 0

    shape = shape_detect.detect(c)

    # draw the contour and center of the shape on the image
    c = (c.astype("float")).astype("int")
    cv_drawContours(image, [c], -1, (0, 255, 0), 2)
    cv_circle(image, (cX, cY), 7, (255, 255, 255), -1)
    cv_putText(image, "center", (cX - 20, cY - 20),
               FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # show the image
    plt.figure(4)
    plt.title(F"Shape detected: {shape}")
    plt.imshow(image)
#    while True:
#        if plt.waitforbuttonpress():
#            break
    while not plt.waitforbuttonpress(): pass
    plt.close(4)

# %%     PLOTTING
# %%

if ENABLE_PLOTTING:

    plt.close("all")

    plt.figure(1)

    fig_1 = plt.subplot(131)
    plt.title("Original image")
    fig_2 = plt.subplot(132)
    plt.title("RGB to GRAY")
    fig_3 = plt.subplot(133)
    plt.title("GRAY to Gaussian Blur")

    fig_1.get_shared_x_axes().join(fig_1, fig_2, fig_3)

    fig_1.imshow(image)
    fig_2.imshow(gray_img, cmap="gray")
    fig_3.imshow(blur_img, cmap="gray")
    plt.show()


    plt.figure(2)
    fig_11 = plt.subplot(421)
    plt.title("Static filtering")
    fig_12 = plt.subplot(422)
    plt.title("Static contorurs")

    fig_21 = plt.subplot(423)
    plt.title("Adaptive filtering")
    fig_22 = plt.subplot(424)
    plt.title("Adaptive contours")

    fig_31 = plt.subplot(425)
    plt.title("OTSU filtering")
    fig_32 = plt.subplot(426)
    plt.title("OTSU contorus")

    fig_41 = plt.subplot(427)
    plt.title("Bilateral filtering")
    fig_42 = plt.subplot(428)
    plt.title("Bilateral contours")

    fig_1.get_shared_x_axes().join(fig_11, fig_12, fig_21, fig_22, fig_31, fig_32, fig_41, fig_42)
    fig_11.imshow(static_img, cmap="gray")
    fig_12.imshow(static_img_contour, cmap="gray")
    fig_21.imshow(adaptive_img, cmap="gray")
    fig_22.imshow(adaptive_img_contour, cmap="gray")
    fig_31.imshow(otsu_img, cmap="gray")
    fig_32.imshow(otsu_img_contour, cmap="gray")
    fig_41.imshow(bilateral_img, cmap="gray")
    fig_42.imshow(bilinear_img_contour, cmap="gray")
    plt.show()

