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
#img_name = "bon_v3.jpg"
img_name = "cropped_bon.jpg"
#img_name = "cropped_bon_2.jpg"

image_path = join(imgs_folder, img_name)

# %%   Classes & Functions
# %%


def edges_detector(img_path, plot_results=False):

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
    #  here mode determines how the input array is extended when the filter
    # overlaps a border.

    kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]),
                               np.array([1, 1, 1])])
    print(kernel_laplace, 'is a laplacian kernel')
    out_l = ndimage.convolve(gray, kernel_laplace, mode='reflect')

    if plot_results:
        plt.close('all')
        ax1 = plt.subplot(131)
        plt.imshow(out_h, cmap='gray', origin='lower')

        plt.subplot(132, sharex=ax1),
        plt.imshow(out_v, cmap='gray', origin='lower')

        plt.subplot(133, sharex=ax1)
        plt.imshow(out_l, cmap='gray', origin='lower')


# %%    Main Section
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
plt.subplot(221), plt.imshow(gray, cmap='gray')
plt.subplot(222), plt.imshow(res_img, cmap='gray')

blur = cv2.GaussianBlur(res_img, (5, 5), 20)
plt.subplot(223), plt.imshow(blur, cmap='gray')

ret, thresh = cv2.threshold(blur, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
plt.subplot(224), plt.imshow(thresh)

# %%   Plotting section
# %%
