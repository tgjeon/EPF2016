"""
Created on Thu Apr 28 20:21:56 2016

@author: tgjeon
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
 
 
rcParams['figure.figsize'] = 15, 6    
    

# Parsing time series data and Read it
#dateparse = lambda dates: pd.datetime.strptime(dates, '%d-%m-%Y')
#data = pd.read_csv("./input/RealMarketPriceDataPT-2010-16.csv", 
#                   parse_dates={'timeline': ['date']}, 
#                   index_col='timeline', date_parser=dateparse)


dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M')
data = pd.read_csv("./input/ElectricityPrice/RealMarketPriceDataPT.csv", 
                   parse_dates={'timeline': ['date', '(UTC)']}, 
                   index_col='timeline', date_parser=dateparse)


genLoadParse = lambda genLoad: pd.datetime.strptime(genLoad, '%Y-%m-%d %H:%M')
genLoad = pd.read_csv("./input/GenLoad/RealDataPT.csv", 
                      parse_dates='date', index_col='date', 
                      date_parser=genLoadParse)
                      
data.index
ts = data['Price'] 
ts.head(10)
                      

test_stationarity(ts)

ts_log = np.log(ts)
plt.plot(ts_log)

moving_avg = pd.rolling_mean(ts_log,12)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(12)

ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)

expwighted_avg = pd.ewma(ts_log, halflife=12)
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')


ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)



# Eliminating Trend and Seasonality
#########################################
# BUG HERE 
#########################################
#ts_log_diff = ts_log - ts_log.shift()
#plt.plot(ts_log_diff)
#
#ts_log_diff.dropna(inplace=True)
#test_stationarity(ts_log_diff)

#from statsmodels.tsa.seasonal import seasonal_decompose
#decomposition = seasonal_decompose(ts_log)
#
#trend = decomposition.trend
#seasonal = decomposition.seasonal
#residual = decomposition.resid
#
#plt.subplot(411)
#plt.plot(ts_log, label='Original')
#plt.legend(loc='best')
#plt.subplot(412)
#plt.plot(trend, label='Trend')
#plt.legend(loc='best')
#plt.subplot(413)
#plt.plot(seasonal,label='Seasonality')
#plt.legend(loc='best')
#plt.subplot(414)
#plt.plot(residual, label='Residuals')
#plt.legend(loc='best')
#plt.tight_layout()
#
#ts_log_decompose = residual
#ts_log_decompose.dropna(inplace=True)
#test_stationarity(ts_log_decompose)
#
#
#
## Forecasting
##ACF and PACF plots:
#
#lag_acf = acf(ts_log_diff, nlags=20)
#lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
#
###Plot ACF: 
#plt.subplot(121) 
#plt.plot(lag_acf)
#plt.axhline(y=0,linestyle='--',color='gray')
#plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.title('Autocorrelation Function')
##Plot PACF:
#plt.subplot(122)
#plt.plot(lag_pacf)
#plt.axhline(y=0,linestyle='--',color='gray')
#plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.title('Partial Autocorrelation Function')
#plt.tight_layout()
#plt.show()
#
#
#
## AR Model
#model = ARIMA(ts_log, order=(2, 1, 0))  
#results_AR = model.fit(disp=-1)  
#RSS = sum((results_AR.fittedvalues-ts_log_diff)**2)
#plt.figure()
#plt.plot(ts_log_diff,  color='blue',label='Original')
#plt.plot(results_AR.fittedvalues, color='red', label='AR Model')
#plt.title("RSS: %0.4f" % RSS)
#
#
#
## MA Model
#model = ARIMA(ts_log, order=(0, 1, 2))  
#results_MA = model.fit(disp=-1)  
#plt.figure()
#plt.plot(ts_log_diff)
#plt.plot(results_MA.fittedvalues, color='red')
#plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))
#
## Combined Model
#model = ARIMA(ts_log, order=(2, 1, 2))  
#results_ARIMA = model.fit(disp=-1)  
#plt.figure()
#plt.plot(ts_log_diff)
#plt.plot(results_ARIMA.fittedvalues, color='red')
#plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
#
#
#predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
#print (predictions_ARIMA_diff.head())
#
#predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
#print (predictions_ARIMA_diff_cumsum.head())
#
#predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
#predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
#predictions_ARIMA_log.head()
#
#predictions_ARIMA = np.exp(predictions_ARIMA_log)
#plt.figure()
#plt.plot(ts)
#plt.plot(predictions_ARIMA)
#plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
#
