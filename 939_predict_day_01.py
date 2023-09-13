#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os, time
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import matplotlib.ticker as mticker

import yfinance as yf
from datetime import date
import datetime

n_future = 1
n_past = 30
past_days = 1.5 * n_past
#print(past_days)
tickers = ['0939.HK']
first_day = date.today() - datetime.timedelta(days=past_days)
#print(first_day)
d1 = int(first_day.strftime("%Y"))
d2 = int(first_day.strftime("%m"))
d3 = int(first_day.strftime("%d"))
start_date = date(d1, d2, d3)
#print(d1, d2, d3)
today = date.today()
d1 = int(today.strftime("%Y"))
d2 = int(today.strftime("%m"))
d3 = int(today.strftime("%d"))
#print(d1, d2, d3)
end_date = date(d1, d2, d3 + 1)
df = yf.download(tickers, start=start_date, end=end_date)  # definere datasættet

item = df[-1:]
if date.today().isoweekday() == 5:
  next_day = pd.Series([date.today() + datetime.timedelta(days=3)])
else:
  if date.today().isoweekday() == 6:
    next_day = pd.Series([date.today() + datetime.timedelta(days=2)])
  else:
    next_day = pd.Series([date.today() + datetime.timedelta(days=1)])

next_day = pd.to_datetime(next_day)
item.index = next_day
#df = df.append(item)
df = pd.concat([df, item], axis= 0)

#print(df)

index_date = df.index.strftime('%Y-%m-%d').tolist()
#print('==========================================================')

scaler = StandardScaler()
scaler = scaler.fit(df)
df = scaler.transform(df)

trainX = []
trainY = []

for i in range(n_past, len(df) - n_future + 1):
  trainX.append(df[i - n_past:i, 0:df.shape[1]])
  trainY.append(df[i + n_future -1:i + n_future, 3])

X, y = np.array(trainX), np.array(trainY)

test_length = -1 * n_future - 1

X_test, y_test = X[test_length:], y[test_length:]
index_date_test = index_date[test_length:]

model_pred = load_model('/home/jovyan/scripts/Open_High_Low_Close_Aclose_Volume_close_30_day-01.h5')

train_predictions = model_pred.predict(X_test).flatten()
actual = y_test.flatten()
predict_result = pd.DataFrame(data={'Date':index_date_test, 'Predictions':train_predictions, 'Actuals':actual})
predict_result.index = pd.to_datetime(predict_result['Date'], format='%Y-%M-%d').dt.strftime('%Y-%M-%d')
predict_result, type(predict_result)

predic_result = model_pred.predict(X_test)
forcast_copies = np.repeat(predic_result, df.shape[1], axis=-1)


y_pred_futue = scaler.inverse_transform(forcast_copies)[:,0]
y_pred_futue.flatten()

forcast_copies = np.repeat(y_test, df.shape[1], axis=-1)
actual= scaler.inverse_transform(forcast_copies)[:,0]
actual.flatten()


present_results = pd.DataFrame(data={'Date':index_date_test, 'Predictions':y_pred_futue.flatten(), 'Actuals':actual.flatten()})

#print(present_results), type(present_results),

print(present_results['Date'][1], present_results['Predictions'][1])
