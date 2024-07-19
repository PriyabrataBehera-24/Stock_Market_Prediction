import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import streamlit as st

# Load data
#importing the stock price of infosys of year 2023 from 2 jan to 29 dec


data = pd.read_csv('INFY_DATA.csv')
fd = pd.read_csv('INFY_DATA.csv')

st.title('Stock price prediction')
st.write(data.describe())


#find the MA(Moving Average) 100, it will work for the 101 row it will find the mean of the first 100 closing stock data
#for the first 100 days no MA
#for 101 it will find mean of previous 100 days closing price
#it is a stock analysis tool used by stock market analysts to figure out if MA200>MA100, then the stock goes uptrend
ma100 = data.Close.rolling(100).mean()

#find the MA(Moving Average) 200, it will work for the 201 row it will find the mean of the first 200 closing stock data
ma200 = data.Close.rolling(200).mean()

#VISUALIZATIONS

st.subheader('Closing Price vs Time chart with 100MV and 200MV')
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(data.Close)
st.pyplot(plt)

data = data['Close'].values.reshape(-1, 1)
# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Split data into train and test sets
train_size = int(len(data_normalized) * 0.8)
test_size = len(data_normalized) - train_size
train_data, test_data = data_normalized[0:train_size, :], data_normalized[train_size:len(data_normalized), :]


# Function to create dataset with look back
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])

    #now convert x_train,y_train in numpy array for LSTM model    
    return np.array(X), np.array(Y)


# Create dataset with look back
look_back = 100
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

select = st.selectbox('Choose your algorithm',['Depth Gated','Peephole','Vanilla','KNN'])
if select=='Depth Gated':
    model_dglstm = Sequential()
    model_dglstm.add(LSTM(50, return_sequences=True, input_shape=(1, look_back), implementation=2))  # Depth Gated LSTM
    model_dglstm.add(LSTM(50))
    model_dglstm.add(Dense(1))
    model_dglstm.compile(loss='mean_squared_error', optimizer='adam')
    model_dglstm.fit(X_train, Y_train, epochs=25, batch_size=1, verbose=2)
    train_predict_dglstm = model_dglstm.predict(X_train)
    test_predict_dglstm = model_dglstm.predict(X_test)
    train_predict_dglstm = scaler.inverse_transform(train_predict_dglstm)
    test_predict_dglstm = scaler.inverse_transform(test_predict_dglstm)
    fig = plt.figure(figsize=(12,6))
    plt.plot(data, label='Actual')
    plt.plot(np.concatenate([train_predict_dglstm, test_predict_dglstm]), label='Depth Gated LSTM Predictions')
    plt.legend()
    plt.show()
    st.pyplot(plt)
elif select=='Peephole':
    # Define and compile Peephole LSTM model
    model_plstm = Sequential()
    model_plstm.add(LSTM(50, return_sequences=True, input_shape=(1, look_back), implementation=1))  # Peephole LSTM
    model_plstm.add(LSTM(50))
    model_plstm.add(Dense(1))
    model_plstm.compile(loss='mean_squared_error', optimizer='adam')
    model_plstm.fit(X_train, Y_train, epochs=25, batch_size=1, verbose=2)
    train_predict_plstm = model_plstm.predict(X_train)
    test_predict_plstm = model_plstm.predict(X_test)
    train_predict_plstm = scaler.inverse_transform(train_predict_plstm)
    test_predict_plstm = scaler.inverse_transform(test_predict_plstm)
    fig = plt.figure(figsize=(12,6))
    plt.plot(data, label='Actual')
    plt.plot(np.concatenate([train_predict_plstm, test_predict_plstm]), label='Vanilla LSTM Predictions')
    plt.legend()
    plt.show()
    st.pyplot(plt)
elif select=='Vanilla':
    # Define and compile Vanilla LSTM model
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, input_shape=(1, look_back)))  # Vanilla LSTM
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')
    model_lstm.fit(X_train, Y_train, epochs=25, batch_size=1, verbose=2)
    train_predict_lstm = model_lstm.predict(X_train)
    test_predict_lstm = model_lstm.predict(X_test)
    train_predict_lstm = scaler.inverse_transform(train_predict_lstm)
    test_predict_lstm = scaler.inverse_transform(test_predict_lstm)
    fig = plt.figure(figsize=(12,6))
    plt.plot(data, label='Actual')
    plt.plot(np.concatenate([train_predict_lstm, test_predict_lstm]), label='Vanilla LSTM Predictions')
    plt.legend()
    plt.show()
    st.pyplot(plt)
else:
    #input features to predict whether customer should buy or sell
    #classification problem,whether I should buy or sell the stock
    fd['Open - Close'] = fd['Open'] - fd['Close']
    fd['High - Low'] = fd['High'] - fd['Low']
    X = fd['Open - Close','High - Low']]
    #X.head(5)
    Y = np.where(fd['Close'].shift(-1)>fd['Close'],1,-1)
    #if i purchase a stock today for 500,i have historical data 
    #if next day it is 600, then it is +1,else -1

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state = 44)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn import neighbors
    from sklearn.metrics import accuracy_score

    #optimal value of k
    #to do that we use gridsearchcv, k -> hyperparamter
    params = {'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
    knn = neighbors.KNeighborsClassifier()
    model = GridSearchCV(knn,params,cv=5)
    model.fit(x_train,y_train)
    accuracy_train = accuracy_score(y_train,model.predict(x_train))
    accuracy_test = accuracy_score(y_test,model.predict(x_test))
    #print(accuracy_train,accuracy_test,sep=' ')
    st.write('ACCURACY TRAIN')
    st.write(accuracy_train)
    st.write('ACCURACY TEST')
    st.write(accuracy_test)



