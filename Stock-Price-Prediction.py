#Import the libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf #Instead of import yfinance as yf you can import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

#Get the stock quote
df = yf.download('AAPL', start='2012-01-01', end='2019-12-17')

#Get the number of rows and columns in data set
df.shape

#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.plot(['Close'])
plt.title('Close Price History')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD $',fontsize=18)
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

#training dataset
#create the scaled training data set
train_data=scaled_data[0:training_data_len , :]
#split the data into x_train and y_train data sets
x_train=[]
y_train=[]
for i in range(60,len(train_data)):
x_train.append(train_data[i-60:i,0])
y_train.append(train_data[i,0])
if i<=61:
print(x_train)
print(y_train)
print()
