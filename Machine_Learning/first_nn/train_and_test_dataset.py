# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:30:51 2019

@author: Mihai
"""

#%% IMPORTS
#%%
import os

from os.path import dirname, join
from numpy import asfarray, asarray, zeros, insert, where
from matplotlib.pyplot import figure, close, plot, stem, xlabel, ylabel, title, imshow

from neural_network_v1 import NeuralNetwork
from git_nn import neuralNetwork as git_net

#%% STATIC DEFINITIONS
#%%
"""
It is required that the structure of the folders to be like the next one:
    - MachineLearning:
        - first_nn:
            - test_data:
                - mnist_test_10.csv
                - mnist_test.csv
            - training_data:
                - mnist_train_100.csv
                - mnist_train.csv
"""

#training_file_name = "first_nn/training_data/mnist_train_100.csv"
training_file_name = "first_nn/training_data/mnist_train.csv"

#testing_file_name = "first_nn/test_data/mnist_test_10.csv"
testing_file_name = "first_nn/test_data/mnist_test.csv"

wih_file_name = "my_net_wih.txt"
who_file_name = "my_net_who.txt"

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
path_to_testing_file = join(path, testing_file_name)

training_data_file = open(path_to_training_file.__str__().replace("\\", "/"),
                          'r')
training_data = training_data_file.readlines()
training_data_file.close()

testing_data_file = open(path_to_testing_file.__str__().replace("\\", "/"),
                          'r')
testing_data = testing_data_file.readlines()
testing_data_file.close()

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3

network = NeuralNetwork(input_nodes=input_nodes,
                        hidden_nodes=hidden_nodes,
                        output_nodes=output_nodes,
                        learning_rate=learning_rate)

# -------- TRAINING THE NETWORK -------------
epochs = 1

for e in range(epochs):
    for record in training_data:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        
        network.train(inputs, targets)
# --------------------------------------------
# --------- TESTING THE NETWORK --------------
correct_result = 0
for record in testing_data:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    
    output = network.query(inputs)
    
    nr = where(output == max(output))[0][0]
    
    if correct_label == nr:
        correct_result += 1
    

# --------------------------------------------

print("Detection rate: {} %".format(correct_result / len(testing_data)*100))

#with open(d.__str__().replace("\\", "/"), 'r') as data_file:
#    data_line = data_file.readline()
#    
#    try:
#        while data_line:
#            print(data_line[0])
#            data_line = data_file.__next__()
#    except StopIteration:
#        print("gata")


#%% DELETE
#%%

del path, training_data_file, path_to_testing_file
del input_nodes, hidden_nodes, output_nodes, learning_rate
del path_to_training_file, testing_file_name, training_file_name