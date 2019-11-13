#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

IMDB RAW DATASET DOWNLOAD LINK: mng.bz/0tIo
Precomputed embeddings from 2014 English Wikipedia:
    https://nlp.stanford.edu/projects/glove,
    -> files from this download has coeffs of a pretrained network so it can be
    used to set them on a freshly defined network
    -> 100d, 500d and so on defines the dimensionality of te pretrained
    embedded net

The mentioned database will be used to demonstrate the advantages of a
pretrained word embedding network (using keras pretrained network)
"""

# %%   Imports
# %%

from os import listdir, getcwd
from os.path import join, isdir, isfile

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences as keras_padding

from numpy import asarray, arange, random, zeros, matrix

from matplotlib.pyplot import close, figure, plot, title, legend, show

# %%   Constans
# %%

DATASET_PATH = "aclImdb/aclImdb"
FILES_NR = 4
WORDS_NR = 100  # no workds to take from a review
TRAINING_SAMPLES = 200  # no samples on which to train
VALIDATE_FILES_NR = 10000  # no files use to validate
MAX_WORDS = 10000  # no words to take from keras tokenizer

FILE_NAME = "glove.6B/glove.6B.100d.txt"
EMBEDDING_DIM = 100

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
                with open(join(dir_name, fname), 'r') as f:
                    texts.append(f.read())
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    return texts, labels


def map_coefs_to_word(file) -> dict:
    '''
    '''
    embeddings_index = {}
    with open(file, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(F"Found {len(embeddings_index)} word vectors")
    return embeddings_index


def get_wrd_embedd_mtx(embedds_idx: dict, word_idx: dict) -> matrix:
    '''
    '''
    global MAX_WORDS
    embedding_mtx = zeros((MAX_WORDS, EMBEDDING_DIM))
    for word, i in word_idx.items():
        if i < MAX_WORDS:
            embedding_vector = embedds_idx.get(word)
            if embedding_vector is not None:
                embedding_mtx[i] = embedding_vector
    return embedding_mtx


def build_keras_model():
    global MAX_WORDS, EMBEDDING_DIM, WORDS_NR
    model = Sequential()
    model.add(Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=WORDS_NR))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    return model

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

embeddings_idx = map_coefs_to_word(join(getcwd(), FILE_NAME))
wrd_embedd_mtx = get_wrd_embedd_mtx(embeddings_idx, word_index)

model = build_keras_model()
model.layers[0].set_weights([wrd_embedd_mtx])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32,
                    validation_data=(x_valid, y_valid))

model.save_weights('pretrained_glove_model_100d.h5')


# %%   Plotting section
# %%

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

close('all')

figure(1)
plot(epochs, acc, 'bo', label='training acc')
plot(epochs, val_acc, 'b', label='validation acc')
legend()

figure(2)
plot(epochs, loss, 'bo', label='training loss')
plot(epochs, val_loss, 'b', label='validation loss')
legend()

show()
