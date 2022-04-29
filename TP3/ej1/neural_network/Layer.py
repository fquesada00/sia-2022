from __future__ import annotations
import numpy as np
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
        print(f"CONNNECT TO {self.weights}")

    def propagate_forward(self, inputs):
        self.excitations = np.dot(self.weights, inputs)
        self.activations = self.activation_function(self.excitations)

        return self.activations

    def propagate_backward(self, upper_weights: np.ndarray, upper_errors: np.ndarray):
        # Leave out the bias unit from the upper layer's weights (first column)
        self.errors = self.df_activation_function(
            self.excitations) * np.dot(upper_weights[:, 1:].T, upper_errors)
            
        return self.errors

    def update_accum_adjustment(self, lower_activations: np.ndarray, learning_rate: float): 
        matrix_lower_activations = np.array([lower_activations])
        accum = learning_rate * \
            np.dot(self.errors, matrix_lower_activations)
        if accum[1] < 0:
            print("negative accum: ",accum)
        self.accum_adjustment += accum


    def update_weights(self):
        self.weights += self.accum_adjustment
        self.accum_adjustment = np.zeros(self.weights.shape)
