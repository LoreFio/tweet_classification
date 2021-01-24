#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:26:20 2020

@author: lauragalvezjimenez
@author: lorenzofiorentini
version 1.1
"""

from sentiment_functions import get_data, sub_dict, save_data, processer,\
    stop_words, get_data_orig, sub_dict_fr, processer_fr, stop_words_fr
import time

'''
PARAMETERS

'''
start_time = time.time()

data_orig = get_data_orig()
trainData = "./raw_data/train_ext.csv".replace("ext", data_orig)
testData = "./raw_data/test_ext.csv".replace("ext", data_orig)
    
'''
LOADING AND SAVING DATA

'''
if data_orig == 'fr':
    X_train, y_train, data_train = get_data(trainData, sub_dict_fr,
                                            processer_fr, stop_words_fr,
                                            "french", test = False)
    X_test, y_test, data_test = get_data(testData, sub_dict_fr, processer_fr,
                                         stop_words_fr, "french", test = True)
else:
    X_train, y_train, data_train = get_data(trainData, sub_dict, processer,
                                            stop_words, test = False)
    X_test, y_test, data_test = get_data(testData, sub_dict, processer,
                                          stop_words, test = True)

save_data(data_orig, X_train, X_test, y_train, y_test)
final_time = time.time()
print("Total time %s seconds" % (final_time - start_time))