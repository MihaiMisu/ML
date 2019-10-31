# -*- coding: utf-8 -*-
'''
    Script made to show a heatmap overlaid on the original picture. A heatmap is a map
which shows what regions from a photo were the most important for the network when it made
the prediction. In this way we have a strating point when we make debug on a certain
image which has been clasified in a wrong way.

'''
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, decode_predictions
from keras.applications.vgg16 import preprocess_input, decode_predictions

from numpy import expand_dims, mean, max as np_max, maximum, uint8, clip

from matplotlib.pyplot import matshow, imshow, close, figure

from cv2 import imread, resize, applyColorMap, COLORMAP_JET

#%%     Constans
#%%


#%%     Classes & Functions
#%%
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

#%%      Main Section
#%%

model = VGG16(weights='imagenet')

img_path = 'elephant.png'
img_path = 'images.png'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
decoded_pred = sorted(decode_predictions(preds)[0], key=lambda entry: entry[2])[-1]

african_elephant_output = model.output[:, 386]
last_conv_layer = model.get_layer('block5_conv3')

grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input],
[pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = mean(conv_layer_output_value, axis=-1)

heatmap = maximum(heatmap, 0)
heatmap /= np_max(heatmap)

img = imread(img_path)
heatmap = resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = uint8(255 * heatmap)
heatmap = applyColorMap(heatmap, COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
superimposed_img = deprocess_image(superimposed_img)
#%%     Plotting section
#%%

close("all")

matshow(heatmap)

figure()
imshow(superimposed_img)
