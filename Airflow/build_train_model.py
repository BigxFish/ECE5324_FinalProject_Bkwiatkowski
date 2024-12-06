import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle
import pandas as pd
import numpy as np
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense
import tempfile

def build_train():

    s3 = S3FileSystem()
    # S3 bucket directory (data warehouse)
    DIR_data = 's3://ece5984-s3-bkwiatkowski/Final_Project/data_warehouse/transformed/'                       # Insert here


    X_train_data = np.load(s3.open('{}/{}'.format(DIR_data, 'X_train_data.pkl')), allow_pickle=True)
    X_test_data = np.load(s3.open('{}/{}'.format(DIR_data, 'X_test_data.pkl')), allow_pickle=True)
    y_train_data = np.load(s3.open('{}/{}'.format(DIR_data, 'y_train_data.pkl')), allow_pickle=True)
    y_test_data = np.load(s3.open('{}/{}'.format(DIR_data, 'y_test_data.pkl')), allow_pickle=True)

    # data dataset model
    # Process the data for LSTM
    trainX_data = np.array(X_train_data)
    testX_data = np.array(X_test_data)
    X_train_data = trainX_data.reshape(X_train_data.shape[0], 1, X_train_data.shape[1])
    X_test_data = testX_data.reshape(X_test_data.shape[0], 1, X_test_data.shape[1])

    # Building the LSTM Model
    lstm_data = Sequential()
    lstm_data.add(LSTM(32, input_shape=(1, trainX_data.shape[1]), activation='relu', return_sequences = False))
    lstm_data.add(Dense(1))
    lstm_data.compile(loss='mean_squared_error', optimizer ='adam')

    # Model Training
    history_data = lstm_data.fit(X_train_data, y_train_data, epochs=25, batch_size=8, verbose=1, shuffle=False, validation_data=(X_test_data, y_test_data))

    # Save model temporarily
    with tempfile.TemporaryDirectory() as tempdir:
        lstm_data.save(f"{tempdir}/lstm_data.h5")
        # Push saved model to S3
        s3.put(f"{tempdir}/lstm_data.h5", f"{DIR_data}/lstm_data.h5")

