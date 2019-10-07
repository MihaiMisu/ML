# -*- coding: utf-8 -*-

#%%
import pandas as pd

from os import getcwd
from os.path import dirname, join
from numpy import array, zeros, where

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

path = dirname(getcwd())

path_to_training_file = join(path, training_file_name)
path_to_testing_file = join(path, testing_file_name)

#%%
data_set = pd.read_csv(path_to_training_file)
X_train = data_set.iloc[:, 1:].values
y_train = data_set.iloc[:, 0].values

data_set = pd.read_csv(path_to_testing_file)
X_test = data_set.iloc[:, 1:].values
y_test = data_set.iloc[:, 0].values

#%%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_expected = zeros((len(y_train), 10)) + 0.01
for train in range(len(y_train)):
    y_expected[train][y_train[train]] = 0.99

#%%
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim=100, kernel_initializer="uniform", activation="sigmoid", input_dim=784))
classifier.add(Dense(output_dim=10, kernel_initializer="uniform", activation="sigmoid"))

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

classifier.fit(X_train, y_expected, batch_size=1, epochs=1)


#%%

y_pred = classifier.predict(X_test)

correct = 0
for pred, expect in zip(y_pred, y_test):
    aux = where(pred == max(pred))[0][0]   
    if aux == expect:
        correct += 1

print("acc = {0:.2f}%".format(correct/len(y_test)*100))

