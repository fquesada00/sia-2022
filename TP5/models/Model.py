from numpy import ndarray

from models.Input import Input
from .Dense import Dense
import numpy as np


class Model():
    def __init__(self, input_layer: Input, output_layer: Dense, name=None):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.layers: list[Dense] = []
        self.name = name

        previous_layer = output_layer

        while previous_layer is not input_layer:
            self.layers.append(previous_layer)
            previous_layer = previous_layer.previous_layer

        self.layers.append(input_layer)

        self.layers.reverse()

    def __call__(self, input_data: ndarray):
        output = input_data

        for layer in self.layers[1:]:
            output = layer.activation(np.dot(layer.weights, output))

        return output

    def flatten_weights(self):
        weights = []

        for layer in self.layers:
            weights.append(layer.weights.flatten())

        return weights

    def total_weights(self):
        number_of_weights = 0
        previous_layer = self.output_layer

        while previous_layer is not self.input_layer:
            number_of_weights += previous_layer.weights.size
            previous_layer = previous_layer.previous_layer

        return number_of_weights

    def unflatten_weights(self, weights: ndarray):
        layer_weights = []

        for layer in self.layers:
            np.append(
                layer_weights, (weights[:layer.weights.size].reshape(layer.weights.shape)))

            weights = weights[layer.weights.size:]

        return layer_weights
