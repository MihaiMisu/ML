# -*- coding: utf-8 -*-

#%%     Importing section
#%%
from numpy import (array, reshape)
from matplotlib.pyplot import (close, figure, plot, stem, subplot, title, 
                               xlabel, ylabel, grid, legend)
from pandas import (read_csv, concat)


#%%     Import the training set
#%%
path_to_training_data_set_file = "Google_Stock_Price_Train.csv"
path_to_testing_data_set_file = "Google_Stock_Price_Test.csv"

train_dataframe = read_csv(path_to_training_data_set_file)
train_dataset = train_dataframe.iloc[:, 1:2].values

test_dataframe = read_csv(path_to_testing_data_set_file)
test_dataset = test_dataframe.iloc[:, 1:2].values

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

# Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


regression = Sequential()

#adding the first layer
regression.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#dropout
regression.add(Dropout(0.2))

regression.add(LSTM(units=50, return_sequences=True))
regression.add(Dropout(0.2))

regression.add(LSTM(units=50, return_sequences=True))
regression.add(Dropout(0.2))

regression.add(LSTM(units=50, return_sequences=False))
regression.add(Dropout(0.2))

#output layer
regression.add(Dense(units=1))

regression.compile(optimizer="adam", loss="mean_squared_error")

regression.fit(X_train, y_train, epochs=100, batch_size=32)

#%%     Setting up the testing data
#%%

total_dataset = concat((train_dataframe["Open"], test_dataframe["Open"]), axis=0)

inputs = total_dataset[len(total_dataset) - len(test_dataset) - 60:].values
inputs = inputs.reshape(-1, 1)  # obtain 80 lines and one column
inputs = scale.transform(inputs)

X_test = [];

for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = array(X_test)

X_test = reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#making the predictions
predicted_values = regression.predict(X_test)
predicted_values = scale.inverse_transform(predicted_values)  # obtain unscaled values

#%%     Viewing the results
#%%

close("all")

plot(test_dataset, "r", label="Real Google stock price")
plot(predicted_values, "b", label="Predicted Google stock price")

title("Google Stock Price Prediction"); xlabel("Time"); ylabel("Stock Value")
legend()
grid()












