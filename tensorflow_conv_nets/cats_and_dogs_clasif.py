# -*- coding: utf-8 -*-

#%%
import time
import matplotlib.pyplot as plt

from os import listdir, walk, getcwd
from os.path import isdir, isfile, join, relpath

from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator

from pprint import pprint
from typing import Generator, Tuple
from gc import collect
from functools import wraps

#%%
#%%

dataset_relative_path = "../P16-Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset"

training_set_path = "../P16-Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset/training_set"
test_set_path = "../P16-Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset/test_set"

cats_test_set_path = "../P16-Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset/test_set/cats"
dogs_test_set_path = "../P16-Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset/test_set/dogs"

cats_training_set_path = "../P16-Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset/training_set/cats"
dogs_training_set_path = "../P16-Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset/training_set/dogs"

single_prediction = "../P16-Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset/single_prediction"

#%%
#%%

class DataSetup:
    
    pass

#%%
#%%

# TODO: repair on the first f.__name__ -> not printing function name
def track_func(f):
    @wraps(f)
    def inner(*args, **kwargs):
        print("Entering: {f.__name__}")
        start_time = time.perf_counter()
        res = f(*args, **kwargs)
        end_time = time.perf_counter()
        print(F"Exiting: {f.__name__} ({end_time - start_time})")
        return res
    return inner

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


# TODO: recursive function to get data from a certain path
def get_dataset_from_path(path):
    pass

@track_func
def build_model():
    print("Building up the model...")
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
    input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    
    pprint(model.summary())
    
    return model


def prepare_jpeg_images(path, split_percentage=0) -> Tuple[Generator, Generator]:
    datagen = ImageDataGenerator(rescale=1./255, 
                                 validation_split=split_percentage)
    train_data = datagen.flow_from_directory(
            path,
            target_size=(150, 150), 
            batch_size=20,
            class_mode='binary',
            subset="training")
    validation_data = datagen.flow_from_directory(
            path,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary',
            subset="validation")
    
    return train_data, validation_data
    
#%%
#%%

train_cats = get_files_name_from_path(cats_training_set_path)
train_dogs = get_files_name_from_path(dogs_training_set_path)

test_cats = get_files_name_from_path(cats_test_set_path)
test_dogs = get_files_name_from_path(dogs_test_set_path)

train_gen, validation_gen = prepare_jpeg_images(training_set_path, 0.5)

model = build_model()
history2 = model.fit_generator(
        train_gen,
        steps_per_epoch=200,
        epochs=25,
        validation_data=validation_gen,
        validation_steps=50)
model.save('cats_and_dogs_small_1.h5')

#%%
#%%

history_of_net_train = {
        "history1": {"details": "Net trained for only 15 epochs",
                     "history": history},
        "history2": {"details": "Net trained for 25 epochs",
                     "history": history2}
    }

acc = history_of_net_train.get("history2", {}).get("history", object).history['acc']
val_acc = history_of_net_train.get("history2", {}).get("history", object).history['val_acc']
loss = history_of_net_train.get("history2", {}).get("history", object).history['loss']
val_loss = history_of_net_train.get("history2", {}).get("history", object).history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#%%
#%%

try:
    del dataset_relative_path
    del cats_test_set_path
    del dogs_test_set_path
    del cats_training_set_path
    del dogs_training_set_path,
    del single_prediction,
    del test_set_path
    del training_set_path
except:
    pass

collect()














