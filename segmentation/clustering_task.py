#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

Script divided in multiple parts as it can be observed: Imports, Constants,
etc. The Main Section part is even more divided - each subsection has to be
run independently by the others to see its result

"""

# %%   Imports
# %%

import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from scipy import ndimage
from sklearn.cluster import KMeans

from os import chdir
from os.path import dirname, abspath, join
chdir(dirname(abspath(__file__)))

# %%   Constans
# %%

imgs_folder = "test_images"

img_name = "1.jpeg"
# img_name = "bon.jpg"
# img_name = "bon_v2.jpg"
img_name = "cropped_bon.jpg"
# img_name = "cropped_bon_2.jpg"

image_path = join(imgs_folder, img_name)

# %%   Classes & Functions
# %%


# %%    Main Section
# %%

image = plt.imread(image_path)
image.shape
plt.imshow(image)

# ----------------------------------------

gray = rgb2gray(image)
plt.imshow(gray, cmap='gray')

# ----------------------------------------

gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 1
    else:
        gray_r[i] = 0
gray = gray_r.reshape(gray.shape[0], gray.shape[1])
plt.imshow(gray, cmap='gray')

# ----------------------------------------

image = plt.imread('edge_detect_image.png')
plt.imshow(image)

# ----------------------------------------

# converting to grayscale
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
plt.close('all')
plt.subplot(131), plt.imshow(out_h, cmap='gray')
plt.subplot(133), plt.imshow(out_v, cmap='gray')

kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]),
                           np.array([1, 1, 1])])
print(kernel_laplace, 'is a laplacian kernel')
out_l = ndimage.convolve(gray, kernel_laplace, mode='reflect')

plt.subplot(132), plt.imshow(out_l, cmap='gray')

# ----------------------------------------
# K-means algorithm to segment an image

pic = plt.imread(image_path)/255
plt.close("all")
plt.subplot(121), plt.imshow(pic)

pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
kmeans = KMeans(n_clusters=5, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]

cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
plt.subplot(122), plt.imshow(cluster_pic)


# %%   Plotting section
# %%
