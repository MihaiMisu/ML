# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:30:51 2019

@author: Mihai-PC
"""

#%% IMPORTS
#%%

from pathlib import Path, PurePath

#%% STATIC DEFINITIONS
#%%

training_file_name = "training_data/mnist_train_100.csv"



#%% MAIN SECTION
#%%

path_to_file = Path(__file__).parents[1]

import os
d = os.path.dirname(os.getcwd())
print(d)
d = os.path.join(d, training_file_name)
print(d.__str__())
#path_to_file = path.(path_to_file, training_file_name)
#
#print(path_to_file.joinpath(training_file_name))
data_file = open(d.__str__().replace("\\", "/"), 'r')