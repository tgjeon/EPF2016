# -*- coding: utf-8 -*-
"""
Created on Wed May  4 21:20:30 2016

@author: tgjeon
"""

import pandas as pd
#import numpy as np
import datetime

from pandas.tseries.offsets import Hour#, Minute


# Parameters
DATA_FOR_EXPERIMENT = datetime.datetime(2016,4,22,23,0,0)  # Competition schedule until 2016-04-22
TRAINING_DURATION_START = datetime.datetime(2015,1,1,0,0,0)       # (2015-01-01:2015-12-31)
TRAINING_DURATION_END = datetime.datetime(2015,12,31,23,0,0)
TEST_DURATION_START = datetime.datetime(2016,1,1,0,0,0)
TEST_DURATION_END = datetime.datetime(2016,4,22,23,0,0)

TRAINING_WINDOW_DAYS = 30
TRAINING_WINDOW_HOURS = 24 * TRAINING_WINDOW_DAYS        # Train for 30 days
PREDICTION_WINDOW_HOURS = 24            # Predict for D+1

MOVING_WINDOW_SIZE = TRAINING_WINDOW_HOURS + PREDICTION_WINDOW_HOURS    # Moving window with 30 days
MOVING_WINDOW_STEP = 24         # Moving window move forward 24h


def evaluation(prediction, label):
    return 0

    
def dataPreparation(rawdata):
    # Cut data until 2016-04-22
    rawdata = rawdata[:DATA_FOR_EXPERIMENT]

    data = pd.DataFrame(columns=range(TRAINING_WINDOW_HOURS))
    label = pd.DataFrame(columns=range(PREDICTION_WINDOW_HOURS))
    
    idx = pd.date_range(start=TRAINING_DURATION_START, periods=MOVING_WINDOW_SIZE, freq='H')    

    data_duration = (TEST_DURATION_END - datetime.timedelta(TRAINING_WINDOW_DAYS)) - TRAINING_DURATION_START
    
    for i in range(data_duration.days):    
        expdata = rawdata.loc[idx]
        
        data_row = (expdata[:TRAINING_WINDOW_HOURS].values).transpose()
        label_row = (expdata[TRAINING_WINDOW_HOURS:].values).transpose()
    
        data = data.append(pd.DataFrame(data=data_row, columns=range(TRAINING_WINDOW_HOURS)), ignore_index=True)
        label = label.append(pd.DataFrame(data=label_row, columns=range(PREDICTION_WINDOW_HOURS)), ignore_index=True)    
    
        idx = idx + Hour(MOVING_WINDOW_STEP)
   
    # Split data into train and test data
    TRAINING_PERIODS = TRAINING_DURATION_END - TRAINING_DURATION_START
    
    train_data = data[:TRAINING_PERIODS.days]
    train_label = label[:TRAINING_PERIODS.days]
    
    test_data = data[TRAINING_PERIODS.days:]
    test_label = label[TRAINING_PERIODS.days:]
    
    return train_data, train_label, test_data, test_label






# Read hourly prices from EPF2016 dataset (2015-01-01:2016-04-27)
dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M')
rawdata = pd.read_csv("./input/ElectricityPrice/RealMarketPriceDataPT.csv", 
                   parse_dates={'timeline': ['date', '(UTC)']}, 
                   index_col='timeline', date_parser=dateparse)
                   

trainX, trainY, testX, testY = dataPreparation(rawdata)

                   




# Comparison groups
# Linear models: ARMA, ARIMA, Kalman filters
# Nonlinear models: NN, SVM, Fuzzy system, Bayesian estimators

