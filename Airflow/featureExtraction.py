import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit


def feature_extract():

    s3 = S3FileSystem()
    # S3 bucket directory (data warehouse)
    DIR_wh = 's3://ece5984-s3-bkwiatkowski/Final_Project/data_warehouse/transformed/'                                 # Insert here
    # Get data from S3 bucket as a pickle file
    data_df = np.load(s3.open('{}/{}'.format(DIR_wh, 'clean_data.pkl')), allow_pickle=True)


    # Set Target Variable
    target_data = pd.DataFrame(data_df['result_spread'])

    # Selecting the Features
    features = ['score_home', 'score_away', 'spread_favorite', 'weather_temperature', 'weather_wind_mph']

    # Scaling dataframe
    scaler = MinMaxScaler()
    feature_transform_data = scaler.fit_transform(data_df[features])
    feature_transform_data = pd.DataFrame(columns=features, data=feature_transform_data, index=data_df.index)

    # Splitting to Training set and Test set
    timesplit = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in timesplit.split(feature_transform_data):
        X_train, X_test = feature_transform_data[:len(train_index)], feature_transform_data[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = target_data[:len(train_index)].values.ravel(), target_data[len(train_index): (len(train_index)+len(test_index))].values.ravel()

    # Push extracted features to data warehouse
    DIR_data = 's3://ece5984-s3-bkwiatkowski/Final_Project/data_warehouse/transformed/'                               # Insert here
    with s3.open('{}/{}'.format(DIR_data, 'X_train_data.pkl'), 'wb') as f:
        f.write(pickle.dumps(X_train))
    with s3.open('{}/{}'.format(DIR_data, 'X_test_data.pkl'), 'wb') as f:
        f.write(pickle.dumps(X_test))
    with s3.open('{}/{}'.format(DIR_data, 'y_train_data.pkl'), 'wb') as f:
        f.write(pickle.dumps(y_train))
    with s3.open('{}/{}'.format(DIR_data, 'y_test_data.pkl'), 'wb') as f:
        f.write(pickle.dumps(y_test))
