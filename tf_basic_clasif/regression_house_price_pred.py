# -*- coding: utf-8 -*-

#%%

import numpy as np

from keras.datasets import boston_housing
from matplotlib.pyplot import (close, figure, plot, subplot, scatter, title, xlabel, ylabel, show)
from numpy import array, reshape, linspace, arange
from keras import models
from keras import layers
from typing import TypeVar

KerasNetModel = TypeVar("NeuralNetModel")

#%%
#%%

def build_model() -> KerasNetModel:
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
    input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


#%%
#%%

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()


mean = train_data.mean(axis=0)
train_data_mean = train_data - mean

std = train_data_mean.std(axis=0)
train_data_std = train_data_mean / std

test_data -= mean
test_data /= std

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
             axis=0)
    partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
             axis=0)

    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)

    all_scores.append(val_mae)

num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
             axis=0)
    partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
             axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

#%%
#%%   PLOTTING STUFF

i = 1

close("all")

figure(1); title("Plotting train data")
subplot(131); scatter(arange(len(train_data[:, i])), train_data[:, i])
subplot(132); scatter(arange(len(train_data[:, i])), train_data_mean[:, i])
subplot(133); scatter(arange(len(train_data[:, i])), train_data_std[:, i])


show()



