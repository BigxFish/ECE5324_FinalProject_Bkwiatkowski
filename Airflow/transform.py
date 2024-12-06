import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle

def transform_data():

    s3 = S3FileSystem()
    # S3 bucket directory (data lake)
    DIR = 's3://ece5984-s3-bkwiatkowski/Final_Project/data_lake'                                    # Insert here
    # Get data from S3 bucket as a pickle file
    raw_data = np.load(s3.open('{}/{}'.format(DIR, 'data.pkl')), allow_pickle=True)     # insert here

    # Dropping rows with NaN in them
    data = raw_data.dropna()

    for col in list(data.columns)[1:3]:
        data = data.drop(data[data[col].values > 50].index)
        data = data.drop(data[data[col].values < -30].index)

    for col in list(data.columns)[5:6]:
        data = data.drop(data[data[col].values > 50].index)
        data = data.drop(data[data[col].values < -40].index)

        # Dropping duplicate rows
    data = data.drop_duplicates()

    # Push cleaned data to S3 bucket warehouse
    DIR_wh = 's3://ece5984-s3-bkwiatkowski/Final_Project/data_warehouse/transformed/'                     # Insert here
    with s3.open('{}/{}'.format(DIR_wh, 'clean_data.pkl'), 'wb') as f:
        f.write(pickle.dumps(data))




