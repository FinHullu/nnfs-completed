import numpy as np
from spiral_data import *
import matplotlib.pyplot as plt

X, y = create_data(100, 5)

np.random.seed(0)

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Act_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Act_SM:
    def forward(self, inputs):
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 1:
            y_true = np.eye(dvalues.shape[1])[y_true]
        self.dinputs = (dvalues - y_true) / samples

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CC(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            cor_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            cor_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        nll = -np.log(cor_confidences)
        return nll

layer1 = Layer(2, 20)
layer2 = Layer(20, 5)

act1 = Act_ReLU()
act2 = Act_SM()

lossf = Loss_CC()

learning_rate = 0.01

for iteration in range(1000000):
    
    layer1.forward(X)
    act1.forward(layer1.output)
    layer2.forward(act1.output)
    act2.forward(layer2.output)

    loss = lossf.calculate(act2.output, y)
    
    if iteration % 1000 == 0:
        accuracy = np.mean(np.argmax(act2.output, axis=1)==y)
        print(f'Iteration {iteration}, loss: {loss}, accuracy: {accuracy}')

    act2.backward(act2.output, y)
    layer2.backward(act2.dinputs)
    act1.backward(layer2.dinputs)
    layer1.backward(act1.dinputs)

    layer1.weights -= learning_rate * layer1.dweights
    layer1.biases -= learning_rate * layer1.dbiases
    layer2.weights -= learning_rate * layer2.dweights
    layer2.biases -= learning_rate * layer2.dbiases

plt.subplot(1, 2, 1)
plt.scatter(X.T[0], X.T[1],c=np.argmax(act2.output, axis=1), cmap='viridis')

plt.subplot(1, 2, 2)
plt.scatter(X.T[0], X.T[1],c=y, cmap='viridis')

plt.show()