# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to randomly split the data into training, validation, and test sets
def random_split_data(X, y, train_percentage=0.7, val_percentage=0.15, test_percentage=0.15):
    # Check if percentages sum up to 1
    total_percentage = train_percentage + val_percentage + test_percentage
    if total_percentage != 1.0:
        raise ValueError("Percentages should sum up to 1.0")

    # Get the total number of examples
    total_examples = len(X)

    # Shuffle the data
    indices = np.arange(total_examples)
    np.random.shuffle(indices)

    # Split data into training, validation, and test sets
    train_size = int(train_percentage * total_examples)
    val_size = int(val_percentage * total_examples)

    train_indices = indices[:train_size]
    remaining_indices = indices[train_size:]

    val_indices = remaining_indices[:val_size]
    test_indices = remaining_indices[val_size:]

    # Create sets
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, y_train, X_val, y_val, X_test, y_test

# Function to initialize neural network parameters
def initialize_parameters(D, neuron_count, K):
    # Initialize weights W1 with small random values.
    W1 = 0.01 * np.random.randn(D, neuron_count)
    
    # Initialize biases b1 with zeros.
    b1 = np.zeros((1, neuron_count))
    
    # Initialize weights W2 with small random values.
    W2 = 0.01 * np.random.randn(neuron_count, K)
    
    # Initialize biases b2 with zeros.
    b2 = np.zeros((1, K))
    
    return W1, b1, W2, b2

# Forward Pass Function
def forward_pass(X, W1, b1, W2, b2):
    # Calculate the first layer output z1 and apply ReLU activation function.
    output1 = np.dot(X, W1) + b1
    hidden = np.maximum(0, output1)  # ReLU activation
    
    # Calculate the second layer output z2.
    finalOutput = np.dot(hidden, W2) + b2
    
    # Apply the softmax function to get probabilities.
    exp_scores = np.exp(finalOutput - np.max(finalOutput, axis=1, keepdims=True))
    probs = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-8)

    return hidden, probs

def calculate_loss(probs, y):
    # Calculate cross-entropy loss for each class and sum them for the total loss.
    num_examples = len(y)
    
    # Add a small epsilon to avoid taking the log of zero
    epsilon = 1e-8
    probs = np.maximum(probs, epsilon)
    
    corect_logprobs = -np.log(probs[range(num_examples), y])
    loss = np.sum(corect_logprobs) / num_examples

    return loss

# Backward Pass and Parameter Update Function
def backward_pass(X, y, hidden, probs, W1, W2, b1, b2, learning_rate):
    # Compute gradients.
    num_examples = len(y)
    dscores = probs.copy()
    dscores[range(num_examples), y] -= 1

    # Compute gradients for the second layer parameters.
    dW2 = np.dot(hidden.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)

    # Compute gradients for the first layer output.
    da1 = np.dot(dscores, W2.T)
    da1[hidden <= 0] = 0  # ReLU activation gradient

    # Compute gradients for the first layer parameters.
    dW1 = np.dot(X.T, da1)
    db1 = np.sum(da1, axis=0, keepdims=True)


    # Update parameters.
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2

# Training Neural Network Function with Early Stopping
def train_neural_network_early_stopping(X_train, y_train, X_val, y_val, W1, b1, W2, b2, learning_rate, epochs,  patience=10):
    # Store losses for plotting.
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(epochs):
        # Forward pass for training set.
        hidden, probs = forward_pass(X_train, W1, b1, W2, b2)

        # Loss calculation for training set.
        train_loss = calculate_loss(probs, y_train)
        train_losses.append(train_loss)

        # Backward pass and parameter update for training set.
        W1, b1, W2, b2 = backward_pass(X_train, y_train, hidden, probs, W1, W2, b1, b2, learning_rate)

        # Forward pass for validation set.
        val_hidden, val_probs = forward_pass(X_val, W1, b1, W2, b2)

        # Loss calculation for validation set.
        val_loss = calculate_loss(val_probs, y_val)
        val_losses.append(val_loss)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count == patience:
            print(f"Early stopping at epoch {epoch + 1} with best validation loss: {best_val_loss}")
            break

    return W1, b1, W2, b2, train_losses, val_losses

# Accuracy Calculation Function
def calculate_accuracy(X, y, W1, b1, W2, b2):
    _, probs = forward_pass(X, W1, b1, W2, b2)
    predicted_classes = np.argmax(probs, axis=1)
    accuracy = np.mean(predicted_classes == y)
    return accuracy

# Load your data from "Stress-Lysis.csv"
data = pd.read_csv("Stress-Lysis.csv")
output1 = data.iloc[:, -1]
features = data.drop(columns=["Stress Level"])

# Splitting the data into training, validation, and test sets
X_train, y_train, X_val, y_val, X_test, y_test = random_split_data(features.values, output1.values)

# Initializing parameters for the neural network
W1, b1, W2, b2 = initialize_parameters(D=features.shape[1], neuron_count=200, K=output1.value_counts().size)


# Training the neural network with early stopping
W1, b1, W2, b2, train_losses, val_losses = train_neural_network_early_stopping(
    X_train, y_train, X_val, y_val, W1, b1, W2, b2, learning_rate=0.001, epochs=200,  patience=100
)

# Plot the training loss curve
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.show()

# Calculate accuracy on the test set
testAccuracy = calculate_accuracy(X_test, y_test, W1, b1, W2, b2)
print("Accuracy on the test set: {:.4%}".format(testAccuracy))
