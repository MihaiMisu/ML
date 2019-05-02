# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Importing the dataset
#dataset = pd.read_csv('Social_Network_Ads.csv')
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#%% Encoding categorical data
"""
    Important step - each string must be encoded to a number format to be 
undestood by the model. This is accomplished by LabelEncoder and OneHotEncoder
class from sklearn.preprocessing module.
**IMPORTANT: the rest of the data remains as it was
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%% Feature Scaling
"""
    Inside the training data can be big numbers and there is no point to keep
the memory occupied without a good reason so a scaling operation is done on
the input data
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%% Fitting classifier to the Training set
# Create your classifier here
"""
    Declare and run the structure of the ANN. For this case it is declared 3 
hidden layers and for each is set an activation function.
"""
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(output_dim=6, init="uniform", activation="relu", input_dim=11))
classifier.add(Dropout(p=0.1))

classifier.add(Dense(output_dim=6, init="uniform", activation="relu"))
classifier.add(Dense(output_dim=1, init="uniform", activation="sigmoid"))

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

#%% Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#%% Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

acc = (cm[0,0] + cm[1,1]) / cm.sum()

#%% Homework

hw_input = np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
hw_input = sc.transform(hw_input)

new_predict = classifier.predict(hw_input)
new_pred = new_predict > 0.5

#%% Evaluation the ANN
# Implement K-Fold cross validation
"""
    K-Fold validation is a technique to find the accurancy and variance of the
defined model to measure it's preformances (don't work as expected).
"""

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, init="uniform", activation="relu", input_dim=11))
    classifier.add(Dense(output_dim=6, init="uniform", activation="relu"))
    classifier.add(Dense(output_dim=1, init="uniform", activation="sigmoid"))    
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return classifier    

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)

mean = accuracies.mean()
variance = accuracies.std()

#%% Tuning the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer="adam"):
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, init="uniform", activation="relu", input_dim=11))
    classifier.add(Dense(output_dim=6, init="uniform", activation="relu"))
    classifier.add(Dense(output_dim=1, init="uniform", activation="sigmoid"))    
    classifier.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return classifier    

classifier = KerasClassifier(build_fn=build_classifier)

optimizable_params = {
        "batch_size": [25, 32],
        "nb_epoch": [100, 500],
        "optimizer": ["adam", "rmsprop"]}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=optimizable_params,
                           scoring="accuracy",
                           cv=10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

#%%
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()