These are the transformation files that uses the .csv file that contained the initial data and gets rid of a few columns and creates another variable used with the neural network.
A pkl file for the data was made before using airflow (i.e. without a batch ingest file) because I was having issues with grabbing the data from the kaggle site.
This way the pkl file is easily grabbed in the s3 bucket.
