import numpy as np
from __future__ import annotations
from .ActivationFunction import ActivationFunction


class Layer:
    def __init__(self, size: int, activation_function: str):
        self.size = size
        self.activation_function = ActivationFunction(activation_function)

    def connect_to(self, lower_layer: Layer):
        self.accum_adjustment = np.zeros(
            (self.size, lower_layer.size))

        self.weights = np.random.rand(
            self.size, lower_layer.size)

    def propagate_forward(self, inputs, biases):
        self.excitations = np.dot(self.weights, inputs) + biases
        self.activations = self.activation_function(self.excitation)

        return self.activation

    def propagate_backward(self, upper_weights: np.ndarray, upper_errors: np.ndarray):
        self.errors = self.df_activation_function(
            self.excitations) * np.dot(upper_weights.T, upper_errors)

        return self.errors

    def update_accum_adjustment(self, lower_activations: np.ndarray, learning_rate: float):
        self.accum_adjustment += learning_rate * \
            np.dot(self.errors, lower_activations.T)

    def update_weights(self):
        self.weights += self.accum_adjustment
        self.accum_adjustment = np.zeros(self.weights.shape)
