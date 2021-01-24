#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:26:20 2020

@author: lauragalvezjimenez
@author: lorenzofiorentini
version 1.1
"""

from keras.models import Model
from keras.layers import MaxPooling1D, Conv1D, Dense
from keras.layers.embeddings import Embedding
from keras.regularizers import l1, l2
from keras.layers import Input
import keras
from sentiment_functions import preparing_X, plot_result, load_data,\
    get_data_orig

'''
PARAMETERS

'''

data_orig = get_data_orig()
num_words = 50000
emb_sz = 128
batch_size = 512
kernel_size = 3
stride_size = 2
n_epochs = 4
n_classes = 3
if data_orig == 'big':
    n_classes = 2    
verbose = 1
validation_split = 0.1
learning_rate = 0.001
lambda_reg = 0.00045
lambda_emb_2 = 0.00045 #0.00015
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

X_train, X_test, maxlen, vocab_size, _ = preparing_X(X_train, X_test, num_words)

deep_inputs = Input(shape = (maxlen,))
embedding_layer = Embedding(vocab_size, emb_sz, embeddings_regularizer = l2(
    lambda_emb_2))(deep_inputs)

Conv1D_layer_1 = Conv1D(emb_sz, kernel_size = kernel_size,
                        strides = stride_size, padding = 'same',
                        activation = 'relu')(embedding_layer)
MaxPooling1D_layer_1 = MaxPooling1D(strides = stride_size, padding = 'valid')(
                                        Conv1D_layer_1)
Conv1D_layer_2 = Conv1D(emb_sz, kernel_size = kernel_size,
                        strides = stride_size, padding = 'same', 
                        activation = 'relu')(MaxPooling1D_layer_1)
MaxPooling1D_layer_2 = MaxPooling1D(strides = stride_size, padding = 'valid')(
                                        Conv1D_layer_2)
dense_layer_1 = Dense(n_classes, activation = 'softmax',
                      kernel_regularizer = l1(lambda_reg))(MaxPooling1D_layer_2)
model = Model(inputs = deep_inputs, outputs = dense_layer_1)

model.compile(loss = loss_str, optimizer = keras.optimizers.Adam(
    lr = learning_rate), metrics = ['acc'])

print(model.summary())

history = model.fit(X_train, y_train, batch_size = batch_size,
                    epochs = n_epochs, verbose = verbose,
                    validation_split = validation_split)

score = model.evaluate(X_test, y_test, verbose = verbose)
y_pred = model.predictX(X_test)

plot_result(score, history)