# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:21:14 2020

@author: lfiorentini
"""

'''
CLASSIFYING

'''
from keras_self_attention import SeqSelfAttention
from keras.preprocessing.sequence import pad_sequences
from sentiment_functions import preprocess, process, sub_dict_fr, processer,\
    stop_words_fr
import pickle
from keras.models import load_model
import argparse

'''
LOADING

'''
parser = argparse.ArgumentParser()
parser.add_argument("-f", help = "list of tweets", nargs="*",
                    default = ['Je suis content'])
tweet_list = list(parser.parse_args().f)

with open('tokenizer_fr.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    handle.close()
    
encoding = 'utf8'
with open("maxlen_fr.txt", 'r', encoding = encoding) as f:
    maxlen = int(f.readlines()[0])
    f.close()

model = load_model('LSTM_trained_fr.h5', custom_objects =
                   SeqSelfAttention.get_custom_objects())

with open('multilabel_binarizer_fr.pickle', 'rb') as handle:
    multilabel_binarizer = pickle.load(handle)
    handle.close()

pre_proc_lst = [preprocess(input_str, sub_dict_fr) for input_str in tweet_list]
proc_lst = [process(input_str, processer, stop_words_fr) for input_str in pre_proc_lst]

X_seq = tokenizer.texts_to_sequences(proc_lst)
X_test = pad_sequences(X_seq, padding = 'post', maxlen = maxlen)
y_pred_test = model.predict(X_test)
y_amax = y_pred_test.argmax(axis = -1)
y_pred_test *= 0
for i in range(len(y_pred_test)):
    y_pred_test[i, y_amax[i]] = 1
result = multilabel_binarizer.inverse_transform(y_pred_test)
result = [el[0] for el in result]
print(result)
