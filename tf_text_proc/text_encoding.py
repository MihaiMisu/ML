#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""

# %%   Imports
# %%

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from numpy import zeros, ndarray
from string import printable

# %%   Constans
# %%

text_sample = ["The cat sat on the mat", "The doc ate my homework"]


# %%   Classes & Functions
# %%

def tokenize_word_from_text_sequence(text: list) -> ndarray:
    '''
    Function to tokenize text input which has to be a list of text sequeces.
    Each sequence is taken and processed WORD by WORD (very important that is
    an word encoding). For each word a new token-integer is generated and used
    to generate later on a tensor which encodes that word.

    return: an array with A x B x C dimensions. A - stands for how many text
    sequences has been in the input list; B - stands for how many unique words
    have been found; C - to be added
    '''
    token_index = {}
    for sample in text:
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1

    max_len = len(token_index)

    res = zeros((len(text_sample), max_len, max(token_index.values()) + 1))
    for i, sample in enumerate(text_sample):
        for j, word in list(enumerate(sample.split()))[:max_len]:
            index = token_index.get(word)
            res[i, j, index] = 1

    return res


def tokenize_char_from_text_sequence(text: list) -> ndarray:
    '''
    Same as above, but at character level
    '''
    token_index = dict(zip(range(1, len(printable) + 1), printable))

    max_len = 50
    res = zeros((len(text), max_len, max(token_index.keys()) + 1))
    for i, sample in enumerate(text):
        for j, char in enumerate(sample):
            index = token_index.get(char)
            res[i, j, index] = 1

    return res


def tokenize_using_keras(text: list, words_nr):
    '''
    Wrapper over keras library to get the tokenized sequence
    '''
    tokenizer = Tokenizer(num_words=words_nr)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)

    one_hot_res = tokenizer.texts_to_matrix(text, mode='binary')

    word_index = tokenizer.word_index
    return word_index


def tokenize_using_hash(text: list):
    '''
    '''
    dimensionality = 1000
    max_len = sum([len(i.split()) for i in text]) - 1

    res = zeros((len(text), max_len, dimensionality))
    for i, sample in enumerate(text):
        for j, word in list(enumerate(sample.split()))[:max_len]:
            index = abs(hash(word)) % dimensionality
            res[i, j, index] = 1
    return res


# %%    Main Section
# %%

word_encoding = tokenize_word_from_text_sequence(text_sample)
char_encoding = tokenize_char_from_text_sequence(text_sample)
word_encoding_2 = tokenize_using_hash(text_sample)


keras_encoding = tokenize_using_keras(text_sample, 1000)

max_features = 10000
max_len = 20
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(max_features, 8, input_length=max_len))  # why 8???

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)

# %%   Plotting section
# %%
