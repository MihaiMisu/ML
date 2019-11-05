#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

IMDB RAW DATASET DOWNLOAD LINK: mng.bz/0tIo

The mentioned database will be used to demonstrate the advantages of a
pretrained word embedding network (using keras pretrained network)
"""

# %%   Imports
# %%

from os import listdir, getcwd
from os.path import join

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences as keras_padding

from numpy import asarray, arange, random

# %%   Constans
# %%

dataset_path = "/aclImdb/aclImdb"


# %%   Classes & Functions
# %%




# %%    Main Section
# %%

current_dir = getcwd()

labels = []  # to be replaced with numpy array
texts = []  # to be replaced with numpy array



max_len = 100  # take first 100 workds from each review
training_samples = 200  # the no reviews taken for training
validation_samples = 10000  # the no reviews taken for validation
max_words = 10000  # no words to take from the dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print(F"Found {len(word_index)} unique tokens")

data = keras_padding(sequences, maxlen=max_len)

labels = asarray(labels)
print(F"Shape of data tensor: {data.shape}")
print(F"Shape of label tensor: {labels.shape}")

indices = arange(data.shape[0])
random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# %%   Plotting section 
# %%


