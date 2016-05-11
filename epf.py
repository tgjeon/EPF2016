# -*- coding: utf-8 -*-
"""
Created on Wed May  4 21:20:30 2016

@author: tgjeon
"""

import pandas as pd
import numpy as np


def evaluation(prediction, label):
    return 0
    
def dataPreparation(data):
    # Cut data until 2016-04-22
    data = data[:DATA_FOR_EXPERIMENT]
                       
    # Make label for future 5-days                    
    labels = data[PREDICTION_HOURS:].values
    data = data[:-PREDICTION_HOURS].values
    
    # Split data into train and test data
    train_data = data[:TRAINING_HOURS]
    train_label = labels[:TRAINING_HOURS]
    
    test_data = data[TRAINING_HOURS:]
    test_label = labels[TRAINING_HOURS:]
    
    return train_data, train_label, test_data, test_label



# Parameters
DATA_FOR_EXPERIMENT = 24 * 478  # Competition schedule until 2016-04-22
PREDICTION_HOURS = 1            # h+1
TRAINING_HOURS = 24 * 365       # (2015-01-01:2015-12-31)
MOVING_WINDOWS_HOURS = 24 * 30  # Moving window with 30 days


# Read hourly prices from EPF2016 dataset (2015-01-01:2016-04-27)
dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M')
rawdata = pd.read_csv("./input/ElectricityPrice/RealMarketPriceDataPT.csv", 
                   parse_dates={'timeline': ['date', '(UTC)']}, 
                   index_col='timeline', date_parser=dateparse)
                   

trainX, trainY, testX, testY = dataPreparation(rawdata)
                   




# Comparison groups
# Linear models: ARMA, ARIMA, Kalman filters
# Nonlinear models: NN, SVM, Fuzzy system, Bayesian estimators

