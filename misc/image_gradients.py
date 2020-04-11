# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 00:51:12 2020

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

from numpy import matrix as np_matrix
from os import chdir, getcwd
from os.path import (join, dirname, abspath)
from cv2 import (imread as cv_imread, resize as cv_resize, cvtColor as cv_cvtColor,
                 adaptiveThreshold as cv_adaptiveThold, threshold as cv_tHold,
                 Laplacian as cv_laplace, Canny as cv_canny,
                 INTER_LINEAR, THRESH_BINARY, ADAPTIVE_THRESH_GAUSSIAN_C,
                 THRESH_OTSU,
                 Sobel as cv_Sobel, CV_64F, COLOR_BGR2RGB, COLOR_RGB2GRAY)

# %%     CONSTANTS
# %%
chdir(dirname(abspath(__file__)))
print(F"Current working directory: {getcwd()}")

IMG_PATH = join(getcwd(), "..", "mak_rcnn_pytorch", "Receipts", "IMG_1167.JPG")
#IMG_PATH = join(getcwd(), "..", "shape_detecting", "shapes_and_colors.jpg")

# %%     FUNCTIONS
# %%


def vertical_lines(img, kernel_size=5):
    return cv_Sobel(img, CV_64F, 1, 0, ksize=kernel_size)


def horizontal_lines(img, kernel_size=5):
    return cv_Sobel(img, CV_64F, 0, 1, ksize=kernel_size)

def laplacian_grad(img):
    return cv_laplace(img, CV_64F)


def range_translation_value(old_range: dict, new_range: dict, value, int_val=False):
    old = old_range["max"] - old_range["min"]
    new = new_range["max"] - new_range["min"]
    if int_val:
        return int((value - old_range["min"]) * new / old + new_range["min"])
    return (value - old_range["min"]) * new / old + new_range["min"]


def matrix_values_translation(matrix, old_range: dict, new_range: dict, int_val=False):
    """
    :param matrix: shall be numpy matrix
    """
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = range_translation_value(old_range, new_range, matrix[i, j], int_val)

# %%     CLASSES
# %%

# %%
# %%


init_img = cv_imread(IMG_PATH)
init_img = cv_resize(init_img, None, fx=1/1.3, fy=1/3, interpolation=INTER_LINEAR)
img = cv_cvtColor(init_img, COLOR_BGR2RGB)
img = cv_cvtColor(img, COLOR_RGB2GRAY)

vertical_lines = np_matrix(vertical_lines(img))
horizontal_lines = np_matrix(horizontal_lines(img))

new_range = {"min": 0, "max": 255}

old_range = {"min": vertical_lines.min(), "max": vertical_lines.max()}
matrix_values_translation(vertical_lines, old_range, new_range, True)

old_range = {"min": horizontal_lines.min(), "max": horizontal_lines.max()}
matrix_values_translation(horizontal_lines, old_range, new_range, True)

combination = ((vertical_lines + horizontal_lines) // 2).astype("uint8")

# APPLY THRESHOLD ON IMAGE - STATIC AND ADAPTIVE
_, static_img = cv_tHold(combination, 127, 255, THRESH_BINARY)
# 5th parameter has to be an odd number
adaptive_img = cv_adaptiveThold(combination, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                                THRESH_BINARY, 119, 5)
# OTSU filtering
_, otsu_img = cv_tHold(combination, 0, 255, THRESH_BINARY+THRESH_OTSU)

# Canny edge detection
original_edges = cv_canny(init_img, 100, 200)
gray_edges = cv_canny(img, 100, 200)
combination_edges = cv_canny(combination, 100, 150)

# %%    PLOTTINGS
# %%
plt.close("all")

# %%
plt.figure(1)

plt.subplot(221)
plt.title("Gray image")
plt.imshow(img, cmap="gray")

plt.subplot(222)
plt.title("Vertical lines filter")
plt.imshow(vertical_lines, cmap="gray")

plt.subplot(223)
plt.title("Horizontal lines filter")
plt.imshow(horizontal_lines, cmap="gray")

plt.subplot(224)
plt.title("Vertical+Horizontal lines filter")
plt.imshow(combination, cmap="gray")

# %%
plt.figure(2)

plt.subplot(221)
plt.title("Gray image")
plt.imshow(img, cmap="gray")

plt.subplot(222)
plt.title("Static th")
plt.imshow(static_img, cmap="gray")

plt.subplot(223)
plt.title("Adaptive th")
plt.imshow(adaptive_img, cmap="gray")

plt.subplot(224)
plt.title("Otsu th")
plt.imshow(otsu_img, cmap="gray")

# %%
plt.figure(3)

plt.subplot(221)
plt.title("Gray image")
plt.imshow(img, cmap="gray")

plt.subplot(222)
plt.title("Canny image")
plt.imshow(gray_edges, cmap="gray")

plt.subplot(223)
plt.title("Combination Canny")
plt.imshow(combination_edges, cmap="gray")

plt.subplot(224)
plt.title("Color Canny")
plt.imshow(init_img, cmap="gray")

plt.show()
