import numpy as np


class Perceptron:

    def __init__(self, activation_function, inputs=None, weights=np.random.rand(3)):
        self.inputs = inputs
        self.weights = weights
        self.activation_function = activation_function

    def z_function(self):
        return self.inputs @ np.transpose(self.weights)

    def update_weights(self, weights):
        self.weights = weights

    def calculate_output(self):
        return self.activation_function(self.z_function())
