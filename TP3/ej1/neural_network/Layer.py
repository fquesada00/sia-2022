from __future__ import annotations
import numpy as np

from .constants import BIAS_NEURON_ACTIVATION
from .ActivationFunction import ActivationFunction


class Layer:

    def __init__(self, size: int, activation_function: str):
        self.size = size
        self.activation_function = ActivationFunction(activation_function)

    def connect_to(self, lower_layer: Layer, bias: float):
        self.accum_adjustment = np.zeros(
            (self.size, lower_layer.size + 1))

        self.weights = np.append(np.full((self.size, 1), bias), np.random.uniform(
            low=-1, high=1, size=(self.size, lower_layer.size)), axis=1)

    def propagate_forward(self, inputs, is_last_layer: bool = False):
        self.excitations = np.dot(self.weights, inputs)

        self.activations = self.activation_function(self.excitations)

        # Add bias unit to layer activations if this is not the last layer
        if not is_last_layer:
            self.activations = np.insert(
                self.activations, 0, BIAS_NEURON_ACTIVATION)

        return self.activations

    def propagate_backward(self, upper_weights: np.ndarray, upper_errors: np.ndarray):
        # Leave out the bias unit from the upper layer's weights (first column)
        self.errors = self.activation_function.derivative(
            self.excitations) * np.dot(upper_weights[:, 1:].T, upper_errors)

        return self.errors

    def update_accum_adjustment(self, lower_activations: np.ndarray, learning_rate: float):
        matrix_lower_activations = np.array([lower_activations])
        accum = learning_rate * \
            np.dot(self.errors, matrix_lower_activations)
        self.accum_adjustment += accum

    def update_weights(self):
        self.weights += self.accum_adjustment
        self.accum_adjustment = np.zeros(self.weights.shape)
