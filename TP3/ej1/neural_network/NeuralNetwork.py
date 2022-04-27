import numpy as np

from typing import List

from TP3.ej1.neural_network.Layer import Layer


class NeuralNetwork:
    def __init__(self, hidden_sizes: list, input_size: int, output_size: int, learning_rate: float, bias: float, activation_function: str, batch_size=1):
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.bias = bias
        self.initialize_layers(input_size, output_size, hidden_sizes)

    def initialize_layers(self, input_size: int, output_size: int, hidden_sizes: list):
        self.layers: List[Layer] = []
        self.layers.append(
            Layer(input_size, self.activation_function, self.learning_rate))

        for layer_size in [*hidden_sizes, output_size]:
            layer = Layer(layer_size, self.activation_function,
                          self.learning_rate)
            layer.connect_to(self.layers[-1])
            self.layers.append(layer)

    def propagate_forward(self, input_data: np.ndarray):
        lower_activation = input_data

        for layer in self.layers[1:]:
            layer_activation = layer.propagate_forward(
                lower_activation, np.full((layer.size), -self.bias))

            lower_activation = layer_activation

        return lower_activation

    def propagate_backward(self, output_error: np.ndarray):
        # Start from the output layer
        upper_weights = self.layers[-1].weights
        upper_errors = output_error

        # Iterate over the hidden layers in reverse order
        for index in reversed(range(1, len(self.layers) - 1)):
            layer_errors = self.layers[index].propagate_backward(
                upper_weights, upper_errors)

            self.layers[index].update_accum_adjustment(
                self.layers[index - 1].activations, self.learning_rate)

            upper_errors = layer_errors
            upper_weights = self.layers[index].weights

    def update_weights(self):
        for layer in self.layers:
            layer.update_weights()

    def train(self, input_dataset: np.ndarray, expected_output: np.ndarray):
        batch_iteration = 0

        # Iterate over the input data (the data set)
        for index, input_data in enumerate(input_dataset):

            output_layer = self.layers[-1]
            output = self.propagate_forward(input_data)

            output_error = output_layer.activation_function.derivative(
                output_layer.excitations) * (expected_output[index] - output)

            self.propagate_backward(output_error)

            batch_iteration += 1

            if batch_iteration == self.batch_size:
                self.update_weights()
                batch_iteration = 0
