#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:26:20 2020

@author: lauragalvezjimenez
@author: lorenzofiorentini
version 1.1
"""

from keras.models import Model
from keras.layers import LSTM, Dense, Bidirectional, dot, Activation, Flatten,\
    TimeDistributed
from keras_self_attention import SeqSelfAttention
from keras.layers.embeddings import Embedding
from keras.regularizers import l1
from keras.layers import Input
from keras.backend import int_shape
import keras
import numpy as np
from sentiment_functions import preparing_X, plot_result, load_data,\
    get_data_orig

'''
PARAMETERS

'''
  
data_orig = get_data_orig()
num_words = 50000
emb_sz = 100 
batch_size = 512
n_neurons = 64
n_epochs = 6
n_classes = 3
if data_orig == 'big':
    n_classes = 2    
verbose = 1
validation_split = 0.1
learning_rate = 0.001
lambda_reg = 0.00045
bidir = True
loss_str = 'binary_crossentropy'

'''
LOADING DATA

'''

X_train, X_test, y_train, y_test = load_data(data_orig)

X_loaded = X_test
Y_loaded = y_test
'''
CODE

'''

X_train, X_test, maxlen, vocab_size, tokenizer = preparing_X(X_train, X_test,
                                                             num_words)

embeddings_dictionary = dict()

glove_file = open('glove.6B/glove.6B.' + str(emb_sz) + 'd.txt',
                  encoding = "utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype = 'float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

not_in_glove = 0
embedding_matrix = np.zeros((vocab_size, emb_sz))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
    else:
        not_in_glove += 1
print('not_in_glove:', not_in_glove, 'total words:', vocab_size)

deep_inputs = Input(shape = (maxlen,))
embedding_layer = Embedding(vocab_size, emb_sz, weights = [embedding_matrix],
                            trainable = False)(deep_inputs)
if bidir:
    LSTM_Layer_1 = Bidirectional(LSTM(n_neurons, return_sequences = True,
                                      kernel_regularizer = l1(lambda_reg)))(
                                          embedding_layer)
else:
    LSTM_Layer_1 = LSTM(n_neurons, return_sequences = True,
                        kernel_regularizer = l1(lambda_reg))(embedding_layer)
SSA_Layer = SeqSelfAttention(attention_activation = 'softmax')(LSTM_Layer_1)
dense_input = int(SSA_Layer.shape[-1]) 
FLAT_Layer = Flatten()(SSA_Layer)
"""
SSA_Layer = dot([LSTM_Layer_1, LSTM_Layer_1],
                axes = [2, 2]) / int_shape(LSTM_Layer_1)[1]
SSA_Layer = Activation('softmax', name = 'attention')(SSA_Layer)
SSA_Layer = dot([SSA_Layer, LSTM_Layer_1], axes = [2, 1])
dense_layer_1 = TimeDistributed(Dense(n_classes, activation = 'softmax',
                      kernel_regularizer = l1(lambda_reg)))(SSA_Layer)
"""

dense_layer_1 = Dense(n_classes, activation = 'softmax',
                      input_shape = (None, dense_input),
                      kernel_regularizer = l1(lambda_reg))(FLAT_Layer)
model = Model(inputs = deep_inputs, outputs = dense_layer_1)

model.compile(loss = loss_str, optimizer = keras.optimizers.Adam(
    lr = learning_rate), metrics = ['acc'])

print(model.summary())

history = model.fit(X_train, y_train, batch_size = batch_size,
                    epochs = n_epochs, verbose = verbose,
                    validation_split = validation_split)

score = model.evaluate(X_test, y_test, verbose = verbose)
y_pred_test = model.predict(X_test)

plot_result(score, history)