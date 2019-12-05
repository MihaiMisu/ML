# -*- coding: utf-8 -*-

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s


"""

# %%   Imports
# %%

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage
import skimage.feature
import skimage.viewer

from scipy import ndimage
from skimage.color import rgb2gray

from os import chdir
from os.path import dirname, abspath, join
chdir(dirname(abspath(__file__)))

# %%   Constans
# %%

imgs_folder = "test_images"

#img_name = "1.jpeg"
#img_name = "bon.jpg"
#img_name = "bon_v2.jpg"
img_name = "bon_v3.jpg"
#img_name = "cropped_bon.jpg"
#img_name = "cropped_bon_2.jpg"

image_path = join(imgs_folder, img_name)

# %%   Classes & Functions
# %%


def adjust(mat):
    not_zero = np.count_nonzero(mat)
    zero = np.count_nonzero(mat == 0)
    if not_zero > zero:
        return 1
    elif not_zero < zero:
        return 0
    return mat[1][1]


def keep_receipt_edges(gray_img):
    for row in range(1, len(gray_img) - 1):
        for col in range(1, len(gray_img[row]) - 1):
            gray_img[row][col] = adjust(gray_img[row-1: row+2, col-1: col+2])
    return gray_img


def binarize_mat(gray_img):
    for row in range(len(gray_img)):
        for col in range(len(gray_img[row])):
            if gray_img[row][col] != 0:
                gray_img[row][col] = 1
    return gray_img


# %%    Main Section
# %%
image = plt.imread(image_path)
gray = rgb2gray(image)

# defining the sobel filters
sobel_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]),
                             np.array([-1, -2, -1])])
print(sobel_horizontal, 'is a kernel for detecting horizontal edges')

sobel_vertical = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]),
                           np.array([-1, 0, 1])])
print(sobel_vertical, 'is a kernel for detecting vertical edges')

out_h = ndimage.convolve(gray, sobel_horizontal, mode='reflect')
out_v = ndimage.convolve(gray, sobel_vertical, mode='reflect')
# here mode determines how the input array is extended when the filter overlaps
# a border.


kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]),
                           np.array([1, 1, 1])])
print(kernel_laplace, 'is a laplacian kernel')
out_l = ndimage.convolve(gray, kernel_laplace, mode='reflect')

# %%

plt.close('all')

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# setting dim of the resize
height = 520
width = 520
dim = (width, height)

res_img = cv2.resize(gray, dim, interpolation=cv2.INTER_LINEAR)

# Checcking the size
print(F"ORIGINAL: {img.shape}")
print(F"RESIZED: {res_img.shape}")

plt.figure()
plt.subplot(241), plt.imshow(gray, cmap='gray')
plt.subplot(242), plt.imshow(res_img, cmap='gray')

blur = cv2.GaussianBlur(res_img, (5, 5), 0)
plt.subplot(243), plt.imshow(blur, cmap='gray')

ret, thresh = cv2.threshold(blur, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
plt.subplot(244), plt.imshow(thresh)

for i in range(len(thresh)):
    for j in range(len(thresh[i])):
        if thresh[i][j] == 255:
            thresh[i][j] = 0
        else:
            thresh[i][j] = 1
for i in range(len(blur)):
    for j in range(len(blur[i])):
        blur[i][j] = thresh[i][j]*blur[i][j]

plt.subplot(245), plt.imshow(blur, cmap='gray')


# Further noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(),
                             255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

plt.figure()
plt.imshow(sure_bg)


# %%

image = skimage.io.imread(fname=image_path, as_gray=True)
plt.imshow(image, cmap='gray')

edges = skimage.feature.canny(
    image=image,
    sigma=0.2,
    low_threshold=0.07,
    high_threshold=0.1,
)
plt.imshow(edges)


def filter_function(image, sigma, threshold):
    masked = image.copy()
    masked[skimage.filters.gaussian(image, sigma=sigma) <= threshold] = 0
    return masked


canny_plugin = skimage.viewer.widgets.Slider(
    name="sigma", low=0.0, high=7.0, value=2.0
)
canny_plugin += skimage.viewer.widgets.Slider(
    name="low_threshold", low=0.0, high=1.0, value=0.1
)
canny_plugin += skimage.viewer.widgets.Slider(
    name="high_threshold", low=0.0, high=1.0, value=0.2
)

viewer = skimage.viewer.ImageViewer(image)
viewer += canny_plugin
viewer.show()


# %%   Plotting section
# %%

plt.close('all')
ax1 = plt.subplot(131)
plt.imshow(out_h, cmap='gray', origin='lower')

plt.subplot(132, sharex=ax1),
plt.imshow(out_v, cmap='gray', origin='lower')

plt.subplot(133, sharex=ax1)
plt.imshow(out_l, cmap='gray', origin='lower')
