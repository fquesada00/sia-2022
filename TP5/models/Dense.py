from .ActivationFunction import ActivationFunction
from .Layer import Layer
import numpy as np


class Dense(Layer):
    def __init__(self, dim, activation='relu', name=None):
        super().__init__(dim)
        self.activation = ActivationFunction(activation)
        self.name = name

    def __call__(self, previous_layer: Layer):
        # Connect
        self.weights = np.random.uniform(-1, 1, (self.dim, previous_layer.dim))
        self.previous_layer = previous_layer

        return self

    def forward(self, input_data, weights: np.ndarray):
        weights = weights.reshape(self.dim, self.previous_layer.dim)
        excitation = np.dot(input_data, weights.T)
        activation = self.activation(excitation)
        return activation

    def update_weights(self, weights: np.ndarray):
        weights = weights.reshape(self.dim, self.previous_layer.dim)
        self.weights = weights

    def load_weights(self, weights: np.ndarray):
        weights = weights.reshape(self.dim, self.previous_layer.dim)
        self.weights = weights
