# -*- coding: utf-8 -*-
"""
Created on Wed May  4 21:20:30 2016

@author: tgjeon
"""

import pandas as pd
import numpy as np

# For sliding_window function
from collections import deque
from itertools import islice


# Parameters
DATA_FOR_EXPERIMENT = 24 * 478  # Competition schedule until 2016-04-22
TRAINING_DURATION = 24 * 365       # (2015-01-01:2015-12-31)
TEST_DURATION = DATA_FOR_EXPERIMENT - TRAINING_DURATION

TRAINING_HOURS = 24 * 30        # Train for 30 days
PREDICTION_HOURS = 24            # Predict for D+1

MOVING_WINDOW_SIZE = TRAINING_HOURS + PREDICTION_HOURS    # Moving window with 30 days
MOVING_WINDOW_STEP = 24         # Moving window move forward 24h


def evaluation(prediction, label):
    return 0

def sliding_window(iterable, size=2, step=1, fillvalue=None):
    if size < 0 or step < 1:
        raise ValueError
    it = iter(iterable)
    q = deque(islice(it, size), maxlen=size)
    if not q:
        return  # empty iterable or size == 0
    q.extend(fillvalue for _ in range(size - len(q)))  # pad to size
    while True:
        yield iter(q)  # iter() to avoid accidental outside modifications
        q.append(next(it))
        q.extend(next(it, fillvalue) for _ in range(step - 1))
        
    
def dataPreparation(rawdata):
    # Cut data until 2016-04-22
    rawdata = rawdata[:DATA_FOR_EXPERIMENT]
    
    it = sliding_window(rawdata['Price'], size=MOVING_WINDOW_SIZE, step=MOVING_WINDOW_STEP)

    data = pd.DataFrame()   
    label = pd.DataFrame()
 
# BUG HERE   
#    for x in it:
#        data.append(x[:TRAINING_HOURS])
#        label.append(x[TRAINING_HOURS:])
#        for yy in x:
#            print (yy)
            
                       
    # Make label for future 5-days                    
#    labels = data[PREDICTION_HOURS:].values
#    data = data[:-PREDICTION_HOURS].values
    
    # Split data into train and test data
#    train_data = data[:TRAINING_HOURS]
#    train_label = labels[:TRAINING_HOURS]
    
#    test_data = data[TRAINING_HOURS:]
 #   test_label = labels[TRAINING_HOURS:]
    
#    return train_data, train_label, test_data, test_label

    return data, label




# Read hourly prices from EPF2016 dataset (2015-01-01:2016-04-27)
dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M')
rawdata = pd.read_csv("./input/ElectricityPrice/RealMarketPriceDataPT.csv", 
                   parse_dates={'timeline': ['date', '(UTC)']}, 
                   index_col='timeline', date_parser=dateparse)
                   

#trainX, trainY, testX, testY = dataPreparation(rawdata)

trainX, trainY = dataPreparation(rawdata)
                   




# Comparison groups
# Linear models: ARMA, ARIMA, Kalman filters
# Nonlinear models: NN, SVM, Fuzzy system, Bayesian estimators

