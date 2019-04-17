# -*- coding: utf-8 -*-

#%% IMPORTS
#%%

import cv2

from matplotlib.pyplot import close as fig_close, figure, subplot, plot, xlabel, ylabel, stem, legend, imshow, title
from os import getcwd
from os.path import dirname, join
from numpy import where, array, ndarray
from typing import Union

from first_nn import neural_network_v1 as nn
#%% STATIC VARIABLES
#%%

relative_path_to_model_wih = "first_nn/coefs_wih.csv"
relative_path_to_model_who = "first_nn/coefs_who.csv"

image_file_name = "1.jpg"
image_file_name = "1_pinta.png"
image_file_name = "2_pinta.png"
image_file_name = "3_pinta.png"
image_file_name = "2.jpg"

height = 28
width = 28

#%% FUNCTIONS
#%%

def display_image_details(image):    
    print("Image details:\n\t- resolution: {}\t\n".format(image_data.shape[:2]))

def adjust_black_to_white_ratio(image: Union[array, ndarray], threshold: float = 100):
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] > threshold:
                image[i][j] = 255
            else:
                image[i][j] = 10
    return image

#%% MAIN
#%%

if __name__ == "__main__":
   
    path = dirname(getcwd())
    relative_path_to_model_wih = join(path, relative_path_to_model_wih)
    relative_path_to_model_who = join(path, relative_path_to_model_who)
    
    image_data = cv2.imread(image_file_name)
    display_image_details(image_data)
    
    grey_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    image_width, image_heigh = grey_image.shape[:2]
    rescaled_image = cv2.resize(grey_image, (width, height), interpolation=cv2.INTER_AREA)
    rescaled_image = adjust_black_to_white_ratio(rescaled_image, 166)
    
    rescaled_image_resized = 255.0 - rescaled_image.reshape(784).astype(float)
    rescaled_image_resized = (rescaled_image_resized / 255.0 * 0.99) + 0.01
    
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    
    learning_rate = 0.3
    
    network = nn.NeuralNetwork(input_nodes=input_nodes,
                 hidden_nodes=hidden_nodes,
                 output_nodes=output_nodes,
                 learning_rate=learning_rate)
    
    network.load_coeffs_from_file(relative_path_to_model_wih,
                                  relative_path_to_model_who)
    
    output = network.query(rescaled_image_resized)
    
    nr = where(output == max(output))[0][0]
    
    #%% PLOT
    #%%
    
    fig_close("all")
    
    figure(1)
    
    subplot(131); imshow(image_data, cmap="brg"); title("Original image")
    subplot(132); imshow(grey_image, cmap="gray"); title("Gray scale image")
    subplot(133); imshow(rescaled_image, cmap="gray"); title("New image")
    