import numpy as np
import tensorflow as tf
import pickle
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Load the training and testing data
X_train_file = 'X_train_data.pkl'
with open(X_train_file, 'rb') as file:
    X_train_data = pickle.load(file)
    
X_test_file = 'X_test_data.pkl'
with open(X_test_file, 'rb') as file:
    X_test_data = pickle.load(file)

y_train_file = 'y_train_data.pkl'
with open(y_train_file, 'rb') as file:
    y_train_data = pickle.load(file)
    
y_test_file = 'y_test_data.pkl'
with open(y_test_file, 'rb') as file:
    y_test_data = pickle.load(file)

# Reshape data for LSTM
trainX_data = np.array(X_train_data)
testX_data = np.array(X_test_data)
X_train_data = trainX_data.reshape(X_train_data.shape[0], 1, X_train_data.shape[1])
X_test_data = testX_data.reshape(X_test_data.shape[0], 1, X_test_data.shape[1])

# Building the LSTM Model
lstm_data = Sequential()
lstm_data.add(LSTM(32, input_shape=(1, trainX_data.shape[1]), activation='relu', return_sequences=False))
lstm_data.add(Dense(1))
lstm_data.compile(loss='mean_squared_error', optimizer='adam')

# Custom Callback to log training MSE for each epoch
class TrainMSECallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.train_mse = []

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(X_train_data, verbose=0)
        mse = mean_squared_error(y_train_data, predictions)
        self.train_mse.append(mse)
        print(f"Epoch {epoch + 1}: Train MSE: {mse}")

# Initialize the callback
train_mse_callback = TrainMSECallback()

'''

# Model Training
history_data = lstm_data.fit(
    X_train_data, y_train_data, 
    epochs=25, 
    batch_size=8, 
    verbose=1, 
    shuffle=False, 
    validation_data=(X_test_data, y_test_data),
    callbacks=[train_mse_callback]
)

# Save the MSE values
train_mse_values = train_mse_callback.train_mse
print("Training MSE over epochs:", train_mse_values)

# Optional: Save the MSE values to a file
with open('train_mse_values.pkl', 'wb') as mse_file:
    pickle.dump(train_mse_values, mse_file)
    
lstm_data.save('lstm_model_vscode.h5')

'''

model = tf.keras.models.load_model('lstm_model_vscode.h5')

predictions = model.predict(X_test_data)

# Calculate errors
mse = mean_squared_error(y_test_data, predictions)
mae = mean_absolute_error(y_test_data, predictions)

# Display the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Optional: Visualize actual vs predicted
import matplotlib.pyplot as plt

sample_size = int(0.02 * len(y_test_data))  # 5% of test data
indices = np.random.choice(len(y_test_data), sample_size, replace=False)  # Random sample without replacement

# Extract the corresponding values
y_test_sample = np.array(y_test_data)[indices]
predictions_sample = predictions[indices]

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(range(sample_size), y_test_sample, label='Actual Values', alpha=0.7, color='blue', marker='o')
plt.scatter(range(sample_size), predictions_sample, label='Predicted Values', alpha=0.7, color='orange', marker='x')
plt.title('Predictions vs Actual Values (5% Sample)')
plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


