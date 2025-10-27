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

#Create a new dataframe with only 'close column'
data = df.filter(['Close'])

#Convert the dataframe to numpy array
dataset = data.values

#Get the numbr of rows to train the model
training_data_len = math.ceil(len(dataset) * .8)

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data

#Training dataset
#Create the scaled training dataset
train_data = scaled_data[0 : training_data_len  ,: ]
#Split the dataset into two parts
x_train = []
y_tarin = []

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
       if i<=61:
           print(x_train)
           print(y_train)
           print()

#convert the x_train and y_train to numpy arrays
x_train , y_train = np.array(x_train), np.array(y_train)
#reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

#Build the LSTM model
model=Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#compile the model
model.compile(optimizer='adam' , loss='mean_squared_error')

#train the model
model.fit(x_train,y_train , batch_size=1, epochs=1)

# Create the testing data set
# Create a new array containing scaled values from index training_data_len-60 to the end
test_data = scaled_data[training_data_len - 60:, :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
x_test.append(test_data[i-60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# The data is already in the correct 3D shape (samples, time steps, features)
# x_test.shape should be (number of test samples, 60, 1)
x_test = np.reshape(x_test, (x_test.shape[0] , x_test.shape[1],1))

#Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#get the root mean square error(RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse

#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

