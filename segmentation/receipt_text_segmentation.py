# -*- coding: utf-8 -*-

# !/usr/bin/env python3
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

import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy import signal

# %%   Constans
# %%

image_name = "1.jpeg"
#image_name = "bon.jpg"
#image_name = "bon_v2.jpg"
image_name = "cropped_bon.jpg"
#image_name = "cropped_bon_2.jpg"

# %%   Classes & Functions
# %%


def range_map(to_map_array, input_start, input_end, output_start, output_end):
    res = np.zeros((len(to_map_array)))
    mapping_fn = interp1d([input_start, input_end], [output_start, output_end])
    for i in range(len(to_map_array)):
        res[i] = mapping_fn(to_map_array[i])
    return res


def signal_amplifier(x, alpha=0.5):
    for i in range(1, len(x)-1):
        x[i] = x[i]*alpha + x[i-1]*(1-alpha)
    return x


def binarize_8bit_gray_img(img):
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i, j] < 100:
                img[i, j] = 0
                continue
            img[i, j] = 255
    return img


# %%    Main Section
# %%

# Seggmentation with integral projection function

image = plt.imread(image_name)
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_img = binarize_8bit_gray_img(gray_img)

lines_sum, columns_sum = gray_img.sum(axis=0), gray_img.sum(axis=1)

lines_sum = lines_sum / max(lines_sum)
lines_sum = np.exp(lines_sum)

new_rows = range_map(lines_sum, min(lines_sum), max(lines_sum), 0,
                     len(gray_img[:, 1])-1)
new_cols = range_map(columns_sum, min(columns_sum), max(columns_sum), 0,
                     len(gray_img[1, :] - 1))

col_axis = np.linspace(0, len(columns_sum)-1, len(columns_sum))


# %%   Plotting section
# %%

plt.close('all')

plt.figure()
plt.imshow(image)

fig, ax = plt.subplots()
ax.imshow(gray_img, cmap='gray')
ax.plot(new_rows, linewidth=2, color='firebrick')
ax.plot(new_cols, col_axis, color='m')
