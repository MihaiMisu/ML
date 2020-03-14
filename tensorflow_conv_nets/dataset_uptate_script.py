#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 11:24:02 2019

@author: mmanole
"""

import os

from shutil import copyfile

no_files_to_copu = 1000
training_set_path = "../P16-Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset/training_set"
dst_path = os.getcwd()
dst_training_folder = "dataset/training_set"
dst_validation_folder = "dataset/validation_set"


def get_file(path):

    for files in sorted(os.listdir(path)):
        yield files

folder = [folder for folder in os.listdir(training_set_path) if not folder.startswith(".")]

for fld in folder:
    src_path = os.path.join(training_set_path, fld)
    cnt_val = 0
    cnt_train = 0
    for file in get_file(src_path):
        if cnt_val < 1001:
            if "cat" in file:
                cmd = F"cp {os.path.join(src_path, file)} {os.path.join(dst_path, dst_validation_folder, 'cats' ,file)}"
#                print(cmd)
                copyfile(os.path.join(src_path, file), os.path.join(dst_path, dst_validation_folder, 'cats' ,file))
#                os.system(cmd)
            elif "dog":
                cmd = F"cp {os.path.join(src_path, file)} {os.path.join(dst_path, dst_validation_folder, 'dogs', file)}"
#                print(cmd)
                copyfile(os.path.join(src_path, file), os.path.join(dst_path, dst_validation_folder, 'dogs', file))
#                os.system(cmd)
            cnt_val += 1
        else:
            if "cat" in file:
                cmd = F"cp {os.path.join(src_path, file)} {os.path.join(dst_path, dst_training_folder, 'cats' ,file)}"
#                print(cmd)
                copyfile(os.path.join(src_path, file), os.path.join(dst_path, dst_training_folder, 'cats' ,file))
#                os.system(cmd)
            elif "dog":
                cmd = F"cp {os.path.join(src_path, file)} {os.path.join(dst_path, dst_training_folder, 'dogs', file)}"
#                print(cmd)
                copyfile(os.path.join(src_path, file), os.path.join(dst_path, dst_training_folder, 'dogs', file))
#                os.system(cmd)
            cnt_train += 1

    print(F"Validation files: {cnt_val}")
    print(F"Training files: {cnt_train}")









