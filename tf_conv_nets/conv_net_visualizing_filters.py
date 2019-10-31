t# -*- coding: utf-8 -*-
'''
This script is made to visualize how a filter look like and what is its activation patters.

    By plotting the result it can be observed that the first layers filters act like edge,
curved lines and many other forms detector. Middle range filters encode simple textures
made from combinations of edge and colors (got from the filters from bellow). As we go to
an upper layer we will see more complex plots. These ones resembles textures found in natural
images like feathers, eyes, leaves etc.
'''

from keras import backend as K
from keras.applications import VGG16

from numpy import clip, random
from matplotlib.pyplot import (close, figure, subplot, imshow)

#%%     Constans
#%%

TRAINING_DATESET_PATH = "dataset/validation_set"
DOG_CLASS, CAT_CLASS = "dogs", "cats"


#%%     Classes & Functions
#%%

# TODO: try to optimize the process
def deprocess_image(img):
    '''
    From an RGB image whose values are between 0 and 1, it's being made an RGB image with values 
    between 0 and 255.
    Channels number remains the same, 3, as the RGB standard says.

    :return the new image 
    '''
    img -= img.mean()
    img /= (img.std() + 1e-5)
    img *= 0.1
    img += 0.5
    img = clip(img, 0, 1)
    img *= 255
    img = clip(img, 0, 255).astype('uint8')
    return img

# TODO: try to optimize the process
def generate_pattern(model, layer_name, filter_index, size=150):
    '''
    TO BE ADDED
    '''
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = random.random((1, size, size, 3)) * 20 + 128.
    step = 1
    for _ in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)

#%%      Main Section
#%%

#%% Single run
model = VGG16(weights='imagenet', include_top=False)

layer_name = "block4_conv1"
filter_index = 1
image_generated = generate_pattern(model, layer_name, filter_index)

#%%     Plotting section
#%% Single image plot

close("all")
figure(size=(5, 5))
imshow(image_generated)

#%% Multi image subplot


layer_name = 'block4_conv1'
size = 32
rows_nr = 8
col_nr = 8

# TODO: find a nother way to plot a single image, not NR_ROW*NR_COL
close("all")
figure()
for i in range(rows_nr):
    for j in range(col_nr):
        print(F"Filter: {(i*col_nr) + j} done.")
        filter_img = generate_pattern(model, layer_name, i + (j * 2), size=size)

#        horizontal_start = i * size
#        horizontal_end = horizontal_start + size
#        vertical_start = j * size
#        vertical_end = vertical_start + size
#        results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

        subplot(rows_nr, col_nr, (i*col_nr) + j + 1)
        imshow(filter_img, interpolation="nearest", aspect='auto')
