# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

#%% Part 1 - Building the CNN
#%%
# Importing the Keras libraries and packages
# used to implement the ANN
from keras.models import Sequential
# used to implement the convolution step; 2D - since we are working with images
# which are 2d objects
from keras.layers import Conv2D
# to add the puling layers
from keras.layers import MaxPooling2D
# to create the 1d array which will be feeded to the ANN
from keras.layers import Flatten
# add fully connected layers
from keras.layers import Dense

#%% Part 2 - initializing the CNN
#%%

classifier = Sequential()

# Step 1 - convolution: added the first layer whose attribution is to do the
# convolution operation. Specified as arguments: the no feature maps, 
# dimensions of the feature to convolve with, input shape of the input data and
# the activation function
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))

# Step 2 - max pooling: reducing the dimension of the feature map => pooling
# layer. In this way we reduce the no flattened layer
classifier.add(MaxPooling2D(pool_size=(2, 2))) 

# Step 3 - flatteneing: taking all the pooled feature maps and reorganise it
# into a single 1d vector which will be feeded to the fully connected network
classifier.add(Flatten())

# Step 4 - build the ANN and input the flattened vector to it
classifier.add(Dense(output_dim=128, activation="relu"))
classifier.add(Dense(output_dim=1, activation="sigmoid"))

# Compiling the CNN: optimizer - adam=stochastic descendent algorithm
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])










# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)