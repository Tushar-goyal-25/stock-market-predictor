#libraries used
import math 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import yfinance as yf

#Get the stock quote
data = yf.download('AAPL', start = '2012-01-01', end='2024-03-30')

#Get the number of rows and columns
# print(data.shape)

def accuracycalc(valid, prediction):
    # Calculate the absolute percentage error
    abs_error = np.abs(valid.values - prediction) / valid.values
    print (abs_error)
    # Calculate the mean absolute percentage error
    mean_abs_error = np.mean(abs_error)
    # Calculate the accuracy
    accuracy = 1 - mean_abs_error
    return accuracy

#Visualise closing price 
# plt.figure(figsize=(16,8))
# plt.title('Close Price')
# plt.plot(data['Close'])
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD', fontsize=18)
# plt.show()
#Create a datafram with close column 
dataf = data.filter(['Close'])
#Convert datafram to numpy array
dataset = dataf.values
#Get number of rows to train the model
training_len = math.ceil(len(dataset)*.8)
# print(training_len)
#Scale the data
scaler = MinMaxScaler(feature_range=(0,1)) #dataset scaled between 0 and 1
scaled_data = scaler.fit_transform(dataset)
#Create the training dataset
#scaled training dataset
train_data = scaled_data[0:training_len, :]
#split between xtrain and ytrain
x_train = [] #independent variables
y_train = [] #dependent variables

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

#convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
#reshape the xtrain dataset
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1))
# print(x_train.shape)
#Building the LSTM Model

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
#Training
model.fit(x_train,y_train, batch_size=1, epochs=1)
#Create a testing dataset
#Create new array with scaled value from index 2404 to 3004
test_data = scaled_data[training_len - 60:, :]
#Create the data set x_test and y_test
x_test = []
y_test = dataset[training_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
#convert the data to a numpy array
x_test = np.array(x_test)

#reshape the data 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#GEt the models predicted value
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Evaluating 
#Get root mean squared error(RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)

#Plot the data
train = data[:training_len]
valid =data[training_len:]
valid['Predictions'] = predictions

#Visualise
plt.figure(figsize=(16,8))
plt.title('Model Predictions')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
# print(valid)

##Predicting the stock for the next day
ndata = data.filter(['Close'])
#get last 60 days values
last_60_days = ndata[-60:].values
#scale data to be values between 0 and 1
last_60_days_sc = scaler.transform(last_60_days)
X_test  =[]
X_test.append(last_60_days_sc)
#Convert X_test to numpy array
X_test = np.array(X_test)
#reshape
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#get the preds
pred_price = model.predict(X_test)
#undo scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)
# print(accuracycalc(valid['Close'], predictions))


