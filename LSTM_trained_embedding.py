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
from keras.regularizers import l1, l2
from keras.layers import Input
from keras.backend import int_shape
import keras
from sentiment_functions import preparing_X, plot_result, load_data,\
    get_data_orig
import tensorflow as tf
'''
PARAMETERS

'''

data_orig = get_data_orig()
print(data_orig)
num_words = 50000
emb_sz = 128 #128 256
batch_size = 512
n_neurons = 64 #64 256
n_epochs = 4
n_classes = 3
if data_orig == 'big' or data_orig == 'fr':
    n_classes = 2
verbose = 1
validation_split = 0.1
learning_rate = 0.001 #0.001 0.0001
lambda_reg = 0.00045
lambda_emb_2 = 0.00045 #0.00015
bidir = True
loss_str = 'binary_crossentropy'

model_file = "LSTM_trained.h5"
language = "english"
if data_orig == 'fr':
    model_file = "LSTM_trained_fr.h5"
    language = "french"

'''
LOADING DATA

'''

X_train, X_test, y_train, y_test = load_data(data_orig)

'''
CODE

'''

X_train, X_test, maxlen, vocab_size, _ = preparing_X(X_train, X_test, num_words,
       language)


deep_inputs = Input(shape = (maxlen,))
embedding_layer = Embedding(vocab_size, emb_sz,
                            embeddings_regularizer = l2(
                                lambda_emb_2))(deep_inputs)
if bidir:
    LSTM_Layer_1 = Bidirectional(LSTM(n_neurons, return_sequences = True,
                                      kernel_regularizer = l1(lambda_reg)))(
                                          embedding_layer)
else:
    LSTM_Layer_1 = LSTM(n_neurons, return_sequences = True,
                        kernel_regularizer = l1(lambda_reg))(embedding_layer)

SSA_Layer = SeqSelfAttention(attention_activation = 'softmax')(LSTM_Layer_1) 
"""
shape = tf.shape( SSA_Layer )
FLAT_Layer = tf.reshape( SSA_Layer, [shape[0] * shape[1], shape[2]] )
"""
dense_input = int(SSA_Layer.shape[-1])
FLAT_Layer = Flatten()(SSA_Layer)
"""
from math import sqrt
print(LSTM_Layer_1.shape)
SSA_Layer = dot([LSTM_Layer_1, LSTM_Layer_1],
                axes = [2, 2]) / sqrt(int_shape(LSTM_Layer_1)[1])
print(SSA_Layer.shape)
SSA_Layer = Activation('softmax', name = 'attention')(SSA_Layer)
SSA_Layer = dot([SSA_Layer, LSTM_Layer_1], axes = [2, 1])
dense_layer_1 = TimeDistributed(Dense(n_classes, activation = 'softmax',
                      kernel_regularizer = l1(lambda_reg)))(SSA_Layer)
print(SSA_Layer.shape)
"""
print(SSA_Layer.shape)
print(FLAT_Layer.shape)
dense_layer_1 = Dense(n_classes, activation = 'softmax',
#                      input_shape = (None, maxlen, dense_input),
                      kernel_regularizer = l1(lambda_reg))(FLAT_Layer)
model = Model(inputs = deep_inputs, outputs = dense_layer_1)

model.compile(loss = loss_str, optimizer = keras.optimizers.Adam(
    lr = learning_rate), metrics = ['acc'])

print(model.summary())

history = model.fit(X_train, y_train, batch_size = batch_size,
                    epochs = n_epochs, verbose = verbose,
                    validation_split = validation_split)
model.save(model_file)
score = model.evaluate(X_test, y_test, verbose = verbose)
y_pred_test = model.predict(X_test)

plot_result(score, history)
