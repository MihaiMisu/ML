# -*- coding: utf-8 -*-

#%%   Imports
#%%

from os import listdir
from os.path import join

from keras import models as keras_models
from keras.models import load_model
from keras.preprocessing import image

from numpy import expand_dims, zeros, clip
from matplotlib.pyplot import close, figure, subplot, plot, stem, title, xlabel, ylabel, grid, imshow, show, matshow

#%%   Constans
#%%

training_dataset_path = "dataset/validation_set"
dog_class, cat_class = "dogs", "cats"

model_to_load = "cats_and_dogs_small_2.h5"

#%%   Classes & Functions
#%%




#%%    Main Section
#%%

model = load_model(model_to_load)
print(model.summary())

#picture = sorted(listdir(join(training_dataset_path, cat_class)))[0]

# process the image: 1. size limit; 2. cast to tensor; 3. add 4th dim (to comply with expected input); 4. Under division values
img = image.load_img(join(training_dataset_path, cat_class, "cat.1700.jpg"), target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)

# Get the layers only from the convolutional part of the network
layer_outputs = [layer.output for layer in model.layers[:8]]
# Build a new model using another keras class which allows intermediate output value
activation_model = keras_models.Model(inputs=model.input, outputs=layer_outputs)

# Get the response of each layer individually
activations = activation_model.predict(img_tensor)
first_layer_output = activations[1]
print(first_layer_output.shape)



#%%   Plotting section 
#%%

close("all")

imshow(img_tensor[0])
matshow(first_layer_output[0, :, :, 7], cmap='viridis')
show()

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    
    size = layer_activation.shape[1]
    
    n_cols = n_features // images_per_row
    display_grid = zeros((size*n_cols, images_per_row*size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_img = layer_activation[0, :, :, col*images_per_row + row]
            channel_img -= channel_img.mean()
            channel_img /= channel_img.std()
            channel_img *= 64
            channel_img += 128
            channel_img = clip(channel_img, 0, 255).astype("uint8")
            display_grid[col*size : (col+1)*size, row*size : (row+1)*size] = channel_img
    
    scale = 1./size
    figure(figsize=(scale*display_grid.shape[1], scale*display_grid.shape[0]))
    title(layer_name)
    grid(False)
    imshow(display_grid, aspect='auto', cmap='viridis')









