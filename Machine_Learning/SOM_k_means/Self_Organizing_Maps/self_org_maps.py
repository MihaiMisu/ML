# -*- coding: utf-8 -*-

#%% Importing the libraries
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

from pylab import bone, pcolor, colorbar, plot, show

#%% Dataset
#%%

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#%% Feature scaling
#%%

sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)


#%% Create the SOM
#%%

som = MiniSom(x=10, y=10, input_len=len(X[0]), learning_rate=0.5, sigma=1.0)

som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

#%% Plot the results
#%%

# Init the window in which the SOM will be displayed
bone()

pcolor(som.distance_map().T)
colorbar()

markers = ["o", "s"]
colors = ["r", "g"]

for index, values in enumerate(X):
    
    w = som.winner(values)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[index]], 
         colors[y[index]], 
         markerfacecolor="None",
         markersize=10,
         markeredgewidth=2)
    
show()


#%% Find the frauds
#%%

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)






