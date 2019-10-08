# -*- coding: utf-8 -*-

#%%

from os import listdir, walk, getcwd
from os.path import isdir, isfile, join, relpath

#%%
#%%

dataset_relative_path = "../P16-Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset"


#%%
#%%

def get_files_nr_from_path(path):
    res = listdir(path)
    return len([file for file in res if isfile(join(path, file)) and not file.startswith(".")])

def get_dirs_nr_from_path(path):
    res = listdir(path)
    return len([directory for directory in res if isdir(join(path, directory))])

def get_files_name_from_path(path):
    res = listdir(path)
    return [file for file in res if isfile(join(path, file)) and not file.startswith(".")]

def get_dirs_name_from_path(path):
    res = listdir(path)
    return [directory for directory in res if isdir(join(path, directory))]



#%%
#%%

dirs = [item for item in listdir(dataset_relative_path) if isdir(join(dataset_relative_path, item))]

for i in dirs:
    sub_dirs_name = get_dirs_name_from_path(join(dataset_relative_path, i))
    for j in sub_dirs_name:
        files_nr = get_files_nr_from_path(join(dataset_relative_path, i, j))
        print(F"In path {join(i, j)} were found {files_nr} files")







