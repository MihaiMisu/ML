# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:30:51 2019

@author: Mihai-PC
"""

#%% IMPORTS
#%%
import os

from os.path import dirname, join
from numpy import asfarray, asarray
from matplotlib.pyplot import figure, close, plot, stem, xlabel, ylabel, title, imshow

#%% STATIC DEFINITIONS
#%%

training_file_name = "training_data/mnist_train_100.csv"


#%% FUNCTIONS
#%%

def mapping(value_to_map, in_min, in_max, out_min, out_max):
    for i in range(len(value_to_map)):
        value_to_map[i] = ((value_to_map[i] - in_min) * (out_max - out_min)/ (in_max - in_min) + out_min)
    return value_to_map

#%% MAIN SECTION
#%%

path = dirname(os.getcwd())
path_to_training_file = join(path, training_file_name)

data_file = open(path_to_training_file.__str__().replace("\\", "/"), 'r')
data_list = data_file.readlines()
data_file.close()

values = data_list[5].split(",")
image_array = asfarray(values[1:]).reshape((28, 28))
#imshow(image_array, cmap="Greys", interpolation="None")

for l in range(len(data_list)):
    data_list[l] = mapping(asfarray(data_list[l].split(",")), 0, 255, 0.01, 1)

#with open(d.__str__().replace("\\", "/"), 'r') as data_file:
#    data_line = data_file.readline()
#    
#    try:
#        while data_line:
#            print(data_line[0])
#            data_line = data_file.__next__()
#    except StopIteration:
#        print("gata")