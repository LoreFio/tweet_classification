#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:24:50 2020

@author: lfiorentini
"""

import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", help = "percentage of the dataset to be used",
                    type = float, default = 0.2)
keep_portion = min([float(parser.parse_args().p), 1])

data = pd.read_csv("./raw_data/french_tweets.csv")
msk = np.random.rand(len(data)) < keep_portion
data = data[msk]
msk = np.random.rand(len(data)) < 0.9
train = data[msk]
test = data[~msk]
train.to_csv('./raw_data/train_fr.csv', sep = ',', index = False,
             encoding = 'utf-8')
test.to_csv('./raw_data/test_fr.csv', sep = ',', index = False,
            encoding = 'utf-8')
