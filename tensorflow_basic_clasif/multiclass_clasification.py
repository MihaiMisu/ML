# -*- coding: utf-8 -*-

#%%
import numpy as np


from keras.datasets import reuters
from keras import models
from keras import layers

import matplotlib.pyplot as plt

#%%
#%%

def vectorize_sequences(sequences, dimension=10000):
#    print(F"Param sequences {sequences}")
#    print(F"len of sequences: {len(sequences)}")
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
#        print(F"i = {i}, len sequence = {len(sequence)}, seq = {sequence}")
#        print()
        results[i, sequence] = 1.
    return results

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
#        results[i, :label] = 0.01
#        results[i, label+1:] = 0.01
    return results

#%%
#%%

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

try:
    del model
except:
    print("No model to delete")

model = models.Sequential()

# input layer
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# 1st hidden layer
model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dropout(0.2))
# 2nd hidden layer
model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dropout(0.1))
# output layer
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=10,
                    batch_size=512,
                    validation_data=(x_val, y_val))

evaluation = model.evaluate(x_test, one_hot_test_labels)
print(F"Loss: {evaluation[0]} --- Acc: {evaluation[1]}")

#%%
#%%

plt.close("all")

plt.figure(1)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(2)
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()
















