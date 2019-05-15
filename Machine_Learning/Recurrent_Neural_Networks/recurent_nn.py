# -*- coding: utf-8 -*-

#%%     Importing section
#%%
from numpy import array, reshape
from matplotlib.pyplot import (close, figure, plot, stem, subplot, title, 
                               xlabel, ylabel)
from pandas import read_csv



#%%     Import the training set
#%%
path_to_training_data_set_file = "Google_Stock_Price_Train.csv"
path_to_testing_data_set_file = "Google_Stock_Price_Test.csv"

train_dataset = read_csv(path_to_training_data_set_file)
train_dataset = train_dataset.iloc[:, 1:2].values



#%%     Build the RNN
#%%

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range=(0, 1))

scaled_training_set = scale.fit_transform(train_dataset) 

# Creating a data structure with 60 timesteps and 1 output
X_train = []; y_train = []

for i in range(60, len(scaled_training_set)):
    X_train.append(scaled_training_set[i-60:i, 0])
    y_train.append(scaled_training_set[i])
X_train, y_train = array(X_train), array(y_train)

X_train = reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



