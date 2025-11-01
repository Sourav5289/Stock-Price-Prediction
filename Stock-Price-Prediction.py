#Import the libraries
import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use ('fivethirtyeight')


#get the stock quote
df = yf.download('AAPL', start='2012-01-01', end='2019-12-17')

#get the number of rows and columns in data set
df.shape


#visualize the closing price histroy
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.show()

#Create a new dataframe with only the 'close column'
data=df.filter(['Close'])

#convert the dataframe to numpy array
dataset=data.values

#Get the number of rows to train the model on
training_data_len=math.ceil(len(dataset) * .8)
training_data_len


#scale the data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
scaled_data