# -*- coding: utf-8 -*-

#%% IMPORTS
#%%

import cv2

from matplotlib.pyplot import close as fig_close, figure, subplot, plot, xlabel, ylabel, stem, legend, imshow, title

#%% STATIC VARIABLES
#%%

image_file_name = "1.jpg"

height = 28
width = 28

#%% FUNCTIONS
#%%

def display_image_details(image):    
    print("Image details:\n\t- resolution: {}\t\n".format(image_data.shape[:2]))

#%% MAIN
#%%

if __name__ == "__main__":
    
    image_data = cv2.imread(image_file_name)
    display_image_details(image_data)
    
    grey_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    image_width, image_heigh = grey_image.shape[:2]
    rescaled_image = cv2.resize(grey_image, (width, height), interpolation=cv2.INTER_AREA)
    
    
    #%% PLOT
    #%%
    
    fig_close("all")
    
    figure(1)
    
    subplot(131); imshow(image_data, cmap="brg"); title("Original image")
    subplot(132); imshow(grey_image, cmap="gray"); title("Gray scale image")
    subplot(133); imshow(rescaled_image, cmap="gray"); title("New image")
    