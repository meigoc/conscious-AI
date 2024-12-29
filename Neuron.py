import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.output = 0

    def activation_function(self, x):
        return 1 / (1 + np.exp(-x))  # Sigmoid https://en.wikipedia.org/wiki/Sigmoid_function

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        self.output = self.activation_function(total)
        return self.output
