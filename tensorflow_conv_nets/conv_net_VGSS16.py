

#%%   Import section
#%%

from keras.applications import VGG16
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator

from matplotlib.pyplot import close, figure, title, grid, xlabel, ylabel, legend, plot, stem, show

from numpy import zeros, reshape

from typing import Tuple, TypeVar

#%%   Static variables
#%%

train_set_path = "dataset/training_set"
validation_set_path = "dataset/validation_set"
test_set_path = "../P16-Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset/test_set"

ConvNet = TypeVar("ConvNet")
DenseNet = TypeVar("DenseNet")



#%%   Functions section
#%%

def extract_features_from_path(conv_base: ConvNet, path: str, samples_nr: int) -> Tuple[list, list]:
    
    batches = 20
    
    features = zeros(shape=(samples_nr, 4, 4, 512))
    labels = zeros(shape=(samples_nr))
    datagen = ImageDataGenerator(rescale=1./255)
    
    data = datagen.flow_from_directory(
            path,
            target_size=(150, 150),
            batch_size=batches,
            class_mode='binary')
    i = 0
    for input_batch, labels_batch in data:
        features_batch = conv_base.predict(input_batch)
        features[i*batches : (i+1)*batches] = features_batch
        labels[i*batches : (i+1)*batches] = labels_batch
        i += 1
        if i*batches >= samples_nr:
            break
    
    return features, labels


def build_dense_model() -> DenseNet:
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crosse',
                  metrics=['acc'])
    print(model.summary())
    
    return model

#%%   Main
#%%


conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
#conv_base.trainable = False
print(conv_base.summary())

train_feat, train_labels = extract_features_from_path(conv_base, train_set_path, 2000)
valid_feat, valid_labels = extract_features_from_path(conv_base, validation_set_path, 1000)
test_feat, test_labels = extract_features_from_path(conv_base, test_set_path, 1000)

train_feat = reshape(train_feat, (2000, 4*4*512))
valid_feat = reshape(valid_feat, (1000, 4*4*512))
test_feat = reshape(test_feat, (1000, 4*4*512))

model = build_dense_model()
history = model.fit(train_feat, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(valid_feat, valid_labels))







#%%   Plotting section
#%%


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plot(epochs, acc, 'bo', label='Training acc')
plot(epochs, val_acc, 'b', label='Validation acc')
title('Training and validation accuracy')
legend()
figure()
plot(epochs, loss, 'bo', label='Training loss')
plot(epochs, val_loss, 'b', label='Validation loss')
title('Training and validation loss')
legend()
show()











