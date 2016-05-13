# -*- coding: utf-8 -*-
"""
Created on Wed May  4 21:20:30 2016

@author: tgjeon
"""

import pandas as pd
import numpy as np
from sklearn import linear_model

import matplotlib.pyplot as plt

import datetime
from pandas.tseries.offsets import Hour#, Minute


# Parameters
DATA_FOR_EXPERIMENT = datetime.datetime(2016,4,22,23,0,0)  # Competition schedule until 2016-04-22
TRAINING_DURATION_START = datetime.datetime(2015,1,1,0,0,0)       # (2015-01-01:2015-12-31)
TRAINING_DURATION_END = datetime.datetime(2016,4,3,23,0,0)
TEST_DURATION_START = datetime.datetime(2016,4,4,0,0,0)
TEST_DURATION_END = datetime.datetime(2016,4,22,23,0,0)

TRAINING_WINDOW_DAYS = 10
TRAINING_WINDOW_HOURS = 24 * TRAINING_WINDOW_DAYS        # Train for 30 days
PREDICTION_WINDOW_HOURS = 24            # Predict for D+1

MOVING_WINDOW_SIZE = TRAINING_WINDOW_HOURS + PREDICTION_WINDOW_HOURS    # Moving window with 30 days
MOVING_WINDOW_STEP = 24         # Moving window move forward 24h

INPUT_LAYER_SIZE = TRAINING_WINDOW_HOURS
OUTPUT_LAYER_SIZE = PREDICTION_WINDOW_HOURS
LEARNING_RATE = 0.001
TRAINING_EPOCHS = 3000
DROPOUT_RATE = 0.8
VALIDATION_SIZE = 0   


# Evaluation with MSE
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
    
    train_data = data[:TRAINING_PERIODS.days].values.astype('float')
    train_label = label[:TRAINING_PERIODS.days].values.astype('float')
    
    test_data = data[TRAINING_PERIODS.days:].values.astype('float')
    test_label = label[TRAINING_PERIODS.days:].values.astype('float')
    
    return train_data, train_label, test_data, test_label



'''
Preprocessing for EPF2016 dataset
'''


# Read hourly prices from EPF2016 dataset (2015-01-01:2016-04-27)
dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M')
rawdata = pd.read_csv("./input/ElectricityPrice/RealMarketPriceDataPT.csv", 
                   parse_dates={'timeline': ['date', '(UTC)']}, 
                   index_col='timeline', date_parser=dateparse)
                   

trainX, trainY, testX, testY = dataPreparation(rawdata)



# Comparison groups
# Linear models: ARMA, ARIMA, Kalman filters
# Nonlinear models: NN, SVM, Fuzzy system, Bayesian estimators

# Linear regression
regr = linear_model.LinearRegression()

# Train the model using training dataset
regr.fit(trainX, trainY)

# Predict using the model 
pred = regr.predict(testX)

# The mean absolute error (MAE)
mae = np.mean(np.abs(pred - testY))

print ("Mean Absolute Error: %.2f" % mae)
print ("Variance score: %.2f" % regr.score(testX, testY))

testY_1D = np.reshape(testY, [1, testY.size])
pred_1D = np.reshape(pred, [1, pred.size])

# Plot outputs
plt.rc('figure', figsize=(15, 7))
plt.plot(range(testY_1D.size), testY_1D[0], color='skyblue', label='Real')
plt.plot(range(pred_1D.size), pred_1D[0], color='orangered', label='Prediction', alpha=0.5, linewidth=2)
plt.xlabel('Hour')
plt.ylabel('Market Price (€/MWh)')
plt.legend(loc='best')

#plt.xticks(())
#plt.yticks(())

plt.savefig('linearRegression.png', dpi=400)
plt.show()


#import tensorflow as tf
#
#def weight_variable(shape):
#    initial = tf.truncated_normal(shape, stddev=0.1)
#    return tf.Variable(initial)
#
## Weight initialization (Xavier's init)
#def weight_xavier_init(n_inputs, n_outputs, uniform=True):
#    if uniform:
#        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
#        return tf.random_uniform_initializer(-init_range, init_range)
#    else:
#        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
#        return tf.truncated_normal_initializer(stddev=stddev)
#
## Bias initialization
#def bias_variable(shape):
#    initial = tf.constant(0.1, shape=shape)
#    return tf.Variable(initial)
#
## 2D convolution
#def conv2d(X, W):
#    return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
#
## Max Pooling
#def max_pool_2x2(X):
#    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
#
#'''
#Create model with 1D CNN
#'''
## Create Input and Output
#X = tf.placeholder('float', shape=[None, INPUT_LAYER_SIZE])    
#Y = tf.placeholder('float', shape=[None, OUTPUT_LAYER_SIZE]) 
#dropout_rate = tf.placeholder("float")
#
## Model Parameters
#W1 = tf.get_variable("W1", shape=[INPUT_LAYER_SIZE, 250], initializer=weight_xavier_init(INPUT_LAYER_SIZE, 250))
#W2 = tf.get_variable("W2", shape=[250, 250], initializer=weight_xavier_init(250, 250))
#W3 = tf.get_variable("W3", shape=[250, 250], initializer=weight_xavier_init(250, 250))
#W4 = tf.get_variable("W4", shape=[250, 250], initializer=weight_xavier_init(250, 250))
#W5 = tf.get_variable("W5", shape=[250, OUTPUT_LAYER_SIZE], initializer=weight_xavier_init(250, OUTPUT_LAYER_SIZE))
#
#B1 = bias_variable([250])
#B2 = bias_variable([250])
#B3 = bias_variable([250])
#B4 = bias_variable([250])
#B5 = bias_variable([OUTPUT_LAYER_SIZE])
#
#
## Construct model
#_L1 = tf.nn.relu(tf.matmul(X, W1) +  B1)
#L1 = tf.nn.dropout(_L1, dropout_rate)
#_L2 = tf.nn.relu(tf.matmul(L1, W2) + B2)
#L2 = tf.nn.dropout(_L2, dropout_rate)
#_L3 = tf.nn.relu(tf.matmul(L2, W3) + B3)
#L3 = tf.nn.dropout(_L3, dropout_rate)
#_L4 = tf.nn.relu(tf.matmul(L3, W4) + B4)
#L4 = tf.nn.dropout(_L4, dropout_rate)
#
#hypothesis = tf.matmul(L4, W5) + B5
#
## Minimize error using cross entropy
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y)) # Softmax loss
#optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost) # Adam Optimizer
#
## Initializing the variables
#init = tf.initialize_all_variables()
#
#with tf.Session() as sess:
#    sess.run(init)
#    
#    # Fit the line
#    for step in range(TRAINING_EPOCHS):
#        sess.run(optimizer, feed_dict={X:trainX, Y:trainY, dropout_rate:DROPOUT_RATE})
#
#        if step % 100 == 0:
#              print (step, sess.run(cost, feed_dict={X:trainX, Y:trainY, dropout_rate:DROPOUT_RATE}), sess.run(W1), sess.run(W2), sess.run(W3), sess.run(W4), sess.run(W5))
#
#        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
#
#        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#        print ("accuracy:", accuracy.eval({X:testX, Y:testY, dropout_rate:1.0}))
