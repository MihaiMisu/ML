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
from os.path import join, isdir, isfile

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences as keras_padding

from numpy import asarray, arange, random

# %%   Constans
# %%

DATASET_PATH = "aclImdb/aclImdb"
FILES_NR = 4
WORDS_NR = 100  # no workds to take from a review
TRAINING_SAMPLES = 200  # no samples on which to train
VALIDATE_FILES_NR = 10000  # no files use to validate
MAX_WORDS = 10000  # no words to take from keras tokenizer
# %%   Classes & Functions
# %%


def grab_files(directory):
    '''
    Generator function used to bring each file a time - memory saving
    '''
    for name in listdir(directory):
        full_path = join(directory, name)
        if isdir(full_path):
            for entry in grab_files(full_path):
                yield entry
        elif isfile(full_path):
            yield full_path
        else:
            print(F'Unidentified name {full_path}. Could be a symbolic link')


def head(iterable, max_iter=10):
    '''
    Function used as a wrapper to stop generating values with the 'iterable'
    argument (assuming that hundrets/thousands/millions of values could be
    generated)
    '''
    first = next(iterable)      # raise exception when depleted

    def head_inner():
        yield first             # yield the extracted first element
        for cnt, el in enumerate(iterable):
            yield el
            if cnt + 2 >= max_iter:  # cnt + 1 to include first
                break
    return head_inner()


def get_data_from_path(path: str):
    '''
    Function to get the review from each text file found on a certain path with
    a certain structure: path shall point to a directory with 2 folders named
    pos and neg representing the review classification. Each review is
    extracted and alongside it the label 1-pos 0-neg
    '''
    labels, texts = [], []

    for label_type in ['neg', 'pos']:
        dir_name = join(path, label_type)
        for fname in listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    return texts, labels

# %%    Main Section
# %%


texts, labels = get_data_from_path(join(getcwd(), "aclImdb/aclImdb/train"))

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = keras_padding(sequences, maxlen=WORDS_NR)
labels = asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = arange(data.shape[0])
random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:TRAINING_SAMPLES]
y_train = labels[:TRAINING_SAMPLES]
x_valid = data[TRAINING_SAMPLES: TRAINING_SAMPLES + VALIDATE_FILES_NR]
y_valid = labels[TRAINING_SAMPLES: TRAINING_SAMPLES + VALIDATE_FILES_NR]


# %%   Plotting section
# %%
