import numpy as np
from perceptron import Perceptron
from activation_function import activation_function
import matplotlib.pyplot as plt

training_set = np.array([
    np.array([1, 0, 0, 0]),
    np.array([1, 0, 1, 0]),
    np.array([1, 1, 0, 0]),
    np.array([1, 1, 1, 1])
])


def compute_error(predicted, real):
    return real - predicted


def train_step(inputs, perceptron, output, training_factor):
    predicted_output = perceptron.calculate_output()
    error = compute_error(predicted_output, output)
    perceptron.update_weights(perceptron.weights + training_factor * error * inputs)


def training():
    inputs = training_set[:, :3]
    outputs = training_set[:, 3:]
    learning_rate = 0.5
    has_changed = True

    perceptron = Perceptron(activation_function)

    while has_changed:
        has_changed = False
        for i in range(len(inputs)):
            perceptron.inputs = inputs[i]
            if perceptron.calculate_output() != outputs[i]:
                train_step(inputs[i], perceptron, outputs[i], learning_rate)
                has_changed = True

    x =
    plt.plot([0, perceptron.inputs[0]], [-perceptron.weights[0] / perceptron.weights[2],
                                 (perceptron.weights[0] - perceptron.inputs[0] * perceptron.weights[1]) /
                               nb
    plt.show()

    return perceptron.weights


training()
# good_weights = training()
#
#
# test_set = np.array([
#     np.array([1, 0.334, 1.56]),
#     np.array([1, 0.22, 0.56]),
#     np.array([1, 1.74, 1.66]),
#     np.array([1, 3.44, 7.99])
# ])
#
#
# def test():
#     perceptron = Perceptron(activation_function, weights=good_weights)
#     for i in range(len(test_set)):
#         perceptron.inputs = test_set[i]
#         output = perceptron.calculate_output()
#         print(test_set[i])
#         print(output)
#
#
# test()