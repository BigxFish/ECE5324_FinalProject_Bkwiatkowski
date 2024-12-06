import matplotlib.pyplot as plt
import pickle

# Load the MSE values if saved in a file
with open('train_mse_values.pkl', 'rb') as mse_file:
    train_mse_values = pickle.load(mse_file)

# Plotting the training MSE over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_mse_values) + 1), train_mse_values, marker='o', label='Training MSE')
plt.title('Training MSE Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.xticks(range(1, len(train_mse_values) + 1))  # Set x-axis ticks to epoch numbers
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

