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

## Second paths used for separate data generator for trainig and validation processes

train_data_gen_path = "dataset/training_set"
validation_data_gen_path = "dataset/validation_set"


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
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
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


@track_func
def build_augumented_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    
    pprint(model.summary())
    
    return model
    

@track_func
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


@track_func
def preapare_augumented_images(training_path, validation_path) -> Tuple[Generator, Generator]:
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    validate_datagen = ImageDataGenerator(
            rescale=1./255)
    
    train_data = train_datagen.flow_from_directory(
            training_path,
            target_size=(150, 150), 
            batch_size=20,
            class_mode='binary')
    validation_data = validate_datagen.flow_from_directory(
            validation_path,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary')
    
    return train_data, validation_data

#%%
#%%

history1, history2, history3 = None, None, None

# All 4 gets can be avoided; not used at the moment
train_cats = get_files_name_from_path(cats_training_set_path)
train_dogs = get_files_name_from_path(dogs_training_set_path)

test_cats = get_files_name_from_path(cats_test_set_path)
test_dogs = get_files_name_from_path(dogs_test_set_path)

train_gen, validation_gen = prepare_jpeg_images(training_set_path, 0.25)
model = build_model()
history1 = model.fit_generator(
        train_gen,
        steps_per_epoch=300,
        epochs=25,
        validation_data=validation_gen,
        validation_steps=100)
model.save('cats_and_dogs_small_1.h5')

train_gen, validation_gen = preapare_augumented_images(train_data_gen_path, validation_data_gen_path)
model = build_augumented_model()
history2 = model.fit_generator(
        train_gen,
        steps_per_epoch=300,
        epochs=25,
        validation_data=validation_gen,
        validation_steps=100)
model.save('cats_and_dogs_small_2.h5')

model = build_augumented_model()
history3 = model.fit_generator(
        train_gen,
        steps_per_epoch=300,
        epochs=80,
        validation_data=validation_gen,
        validation_steps=100)
model.save('cats_and_dogs_small_3.h5')

history_of_net_train = {
        "history1": {"details": "Net trained for only 15 epochs",
                     "history": history1},
        "history2": {"details": "Net trained for 25 epochs with augmented data",
                     "history": history2},
        "history3": {"details": "Net trained for 80 epochs with augmented data",
                     "history": history3}
    }

#%%
#%%

acc = history_of_net_train.get("history3", {}).get("history", object).history['acc']
val_acc = history_of_net_train.get("history3", {}).get("history", object).history['val_acc']
loss = history_of_net_train.get("history3", {}).get("history", object).history['loss']
val_loss = history_of_net_train.get("history3", {}).get("history", object).history['val_loss']
epochs = range(1, len(acc) + 1)

#acc = [0.5403, 0.5830, 0.6123, 0.613, 0.6635, 0.6723, 0.6782, 0.6842, 0.6990, 0.7055, 0.7077, 0.7192, 0.7149, 0.7258, 0.7430, 0.7395, 0.7332, 0.7455, 0.7467, 0.7505, 0.7552, 0.7638, 0.7515, 0.761, 0.7635, 0.7643, 0.7688, 0.78, 0.7867, 0.7717, 0.784, 0.7713, 0.7903, 0.7808, 0.7918, 0.7867, 0.7860, 0.7943, 0.7892, 0.7922, 0.8110, 0.7977, 0.8030, 0.8042, 0.8017, 0.8095, 0.8065, 0.8065, 0.8040, 0.8132, 0.8073, 0.8175, 0.8173, 0.8172, 0.8183, 0.8203, 0.8220, 0.8203, 0.8188, 0.8258, 0.8317, 0.8273, 0.8290, 0.8297, 0.8317, 0.8330, 0.8320, 0.8252, 0.8325, 0.8323, 0.8295, 0.8382, 0.8377, 0.8407, 0.8378, 0.8362, 0.8368, 0.8362, 0.8367, 0.8377]
#val_acc = [0.5755, 0.5755, 0.5185, 0.6686, 0.6970, 0.6910, 0.7090, 0.7105, 0.7065, 0.7525, 0.7260, 0.7360, 0.7360, 0.7545, 0.7615, 0.777, 0.783, 0.7755, 0.7185, 0.7765, 0.7765, 0.7870, 0.787, 0.791, 0.7895, 0.7695, 0.8060, 0.782, 0.8015, 0.798, 0.738, 0.8050, 0.7815, 0.7655, 0.79, 0.8095, 0.8295, 0.8120, 0.836, 0.8415, 0.836, 0.8325, 0.7760, 0.8070, 0.8380, 0.846, 0.836, 0.8425, 0.8360, 0.8445, 0.84, 0.842, 0.848, 0.839, 0.844, 0.8415, 0.8450, 0.8255, 0.8465, 0.8165, 0.8305, 0.8435, 0.8535, 0.8530, 0.8655, 0.8250, 0.863, 0.7885, 0.846, 0.8645, 0.823, 0.8315, 0.857, 0.8605, 0.796, 0.8515, 0.8715, 0.8525, 0.843, 0.87]
#
#loss = [0.6712, 0.6544, 0.6279, 0.6151, 0.6020, 0.5874, 0.5911, 0.5711, 0.5629, 0.5567, 0.5501, 0.5541, 0.5387, 0.5287, 0.5274, 0.5267, 0.5521, 0.5148, 0.5024, 0.5043, 0.5991, 0.5002, 0.4925, 0.4887, 0.4908, 0.4863, 0.4823, 0.4844, 0.4707, 0.4636, 0.4781, 0.4567, 0.4625, 0.4452, 0.4556, 0.4553, 0.446, 0.4479, 0.445, 0.4261, 0.4331, 0.4361, 0.4341, 0.4381, 0.4252, 0.4288, 0.4196, 0.4244, 0.4134, 0.4242, 0.4086, 0.4087, 0.4090, 0.4032, 0.4028, 0.4003, 0.4068, 0.4127, 0.399, 0.3824, 0.3995, 0.381, 0.3864, 0.387, 0.3879, 0.3843, 0.3943, 0.3809, 0.3914, 0.397, 0.3755, 0.3763, 0.3758, 0.3753, 0.3715, 0.3662, 0.3717, 0.3709, 0.3743, 0.3743]
#val_loss = [0.6748, 0.6704, 0.7885, 0.5909, 0.5775, 0.5843, 0.5546, 0.5568, 0.5679, 0.5197, 0.5419, 0.5369, 0.4942, 0.51084, 0.473, 0.4706, 0.4752, 0.553, 0.4723, 0.4597, 0.4652, 0.4547, 0.4437, 0.467, 0.4998, 0.428, 0.4817, 0.441, 0.4359, 0.5628, 0.4432, 0.4481, 0.5265, 0.4613, 0.5166, 0.3818, 0.4151, 0.389, 0.3705, 0.3831, 0.4030, 0.5073, 0.4340, 0.3777, 0.3595, 0.3866, 0.3808, 0.3778, 0.3784, 0.3965, 0.3893, 0.3633, 0.3825, 0.3624, 0.3676, 0.3809, 0.4209, 0.37171, 0.4406, 0.4236, 0.3999, 0.361, 0.3908, 0.3341, 0.4573, 0.3403, 0.5713, 0.3945, 0.3327, 0.4095, 0.4097, 0.3658, 0.341, 0.4884, 0.3943, 0.3202, 0.3428, 0.3977, 0.3581, 0.3581]

plt.close("all")
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














