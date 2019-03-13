# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:30:51 2019

@author: Mihai-PC
"""

#%% IMPORTS
#%%
import os
from os.path import dirname, join
#%% STATIC DEFINITIONS
#%%

training_file_name = "training_data/mnist_train_100.csv"


#%% MAIN SECTION
#%%

d = dirname(os.getcwd())
d = join(d, training_file_name)

data_file = open(d.__str__().replace("\\", "/"), 'r')