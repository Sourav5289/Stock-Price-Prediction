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
