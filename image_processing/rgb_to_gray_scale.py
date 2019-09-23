# -*- coding: utf-8 -*-

#%% IMPORTS
#%%

import cv2

from matplotlib.pyplot import close as fig_close, figure, subplot, plot, xlabel, ylabel, stem, legend, imshow, title
import matplotlib._png as matlib_png
from matplotlib import image as matlib_img
from numpy import array, zeros


#%% STATIC VARIABLES
#%%

image_file_name = "220px-Lenna.png"
image_file_name = "lena512color.tiff"

#%% IMAGE PROCESSING APIs
#%%

# TODO: Complete with docstring
def rgb_to_gray(image, open_cv_flag: bool = False, r_coef: float = 0.3, g_coef: float = 0.59, b_coef: float = 0.11) -> array:
    """
    """
    
    if open_cv_flag:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_img
    
    gray_img = zeros((len(image), len(image[0])))
    
    for x_axis in range(len(image)):
        
        
        
        for y_axis in range(len(image[x_axis])):
            gray_img[x_axis][y_axis] = r_coef*image_data[x_axis][y_axis][0] +\
                                        g_coef*image_data[x_axis][y_axis][1] +\
                                        b_coef*image_data[x_axis][y_axis][2]
    
    return gray_img
    
#%% MAIN
#%%

if __name__ == "__main__":

    image_data = cv2.imread(image_file_name)
    
    open_cv_gray = rgb_to_gray(image_data, open_cv_flag=True)
    my_gray = rgb_to_gray(image_data)

    cv2.imwrite("opencv_new_img.png", open_cv_gray)
    cv2.imwrite("my_img.png", my_gray)
    #%% PLOT
    #%%
    
    fig_close("all")
    
    figure(1)
    
    subplot(131); imshow(image_data, cmap="brg"); title("Original image")
    subplot(132); imshow(open_cv_gray, cmap="gray"); title("OpenCV image processed")
    subplot(133); imshow(my_gray, cmap="gray"); title("My image")
    



























