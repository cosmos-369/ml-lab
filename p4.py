import numpy as np

#  Simple Dataset: [Hours Studied, Previous Test Score]
X = np.array([
    [2, 80],  # Student 1: 2 hours studied, 80 previous score
    [4, 90],  # Student 2: 4 hours studied, 90 previous score
    [6, 75],  # Student 3: 6 hours studied, 75 previous score
    [8, 85]   # Student 4: 8 hours studied, 85 previous score
], dtype=float)

# Target Output (Final Test Score)
y = np.array([
    [82],  # Actual final score of Student 1
    [94],  # Actual final score of Student 2
    [78],  # Actual final score of Student 3
    [88]   # Actual final score of Student 4
], dtype=float)

#  Normalize the dataset (Feature Scaling)
X = X / np.amax(X, axis=0)  # Normalize input
y = y / 100  # Normalize output (since scores are out of 100)

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)

#  Neural Network Parameters
epochs = 7000  # Number of training iterations
learning_rate = 0.1  # Step size for weight updates
input_neurons = 2  # Two input features
hidden_neurons = 3  # Three neurons in the hidden layer
output_neurons = 1  # One output neuron

#  Initialize weights and biases randomly
wh = np.random.uniform(size=(input_neurons, hidden_neurons))  # Input to Hidden Layer Weights
bh = np.random.uniform(size=(1, hidden_neurons))  # Hidden Layer Bias
wout = np.random.uniform(size=(hidden_neurons, output_neurons))  # Hidden to Output Layer Weights
bout = np.random.uniform(size=(1, output_neurons))  # Output Layer Bias

#  Training the Network
for i in range(epochs):
    #  Forward Propagation
    hinp1 = np.dot(X, wh) + bh  # Input to Hidden Layer
    hlayer_act = sigmoid(hinp1)  # Activation function applied to hidden layer
    outinp1 = np.dot(hlayer_act, wout) + bout  # Hidden to Output Layer
    output = sigmoid(outinp1)  # Activation function applied to output

    #  Backpropagation
    error = y - output  # Calculate error (difference between expected and predicted)
    output_gradient = derivatives_sigmoid(output)
    d_output = error * output_gradient  # Gradient of Output Layer

    error_hidden = d_output.dot(wout.T)  # Backpropagate the error to hidden layer
    hidden_gradient = derivatives_sigmoid(hlayer_act)
    d_hidden_layer = error_hidden * hidden_gradient  # Gradient of Hidden Layer

    # Update Weights and Biases
    wout += hlayer_act.T.dot(d_output) * learning_rate
    bout += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    wh += X.T.dot(d_hidden_layer) * learning_rate
    bh += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

#  Final Output After Training
print("\n Final Results")
print("Input:\n", X)
print("Actual Output:\n", y)
print("Predicted Output:\n", output)
