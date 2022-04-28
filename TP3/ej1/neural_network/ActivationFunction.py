import numpy as np
from enum import Enum


class ActivationFunction():
    LOGISTIC = "logistic"
    TANH = "tanh"
    RELU = "relu"
    STEP = "step"
    IDENTITY = "identity"
    # SOFTMAX = "softmax"

    def __init__(self, name):
        self.function = self.__get_function__(name)
        self._derivative = self.__get_df_function__(name)

    def __str__(self):
        return self.name

    def __get_function__(self):
        if self.name == ActivationFunction.LOGISTIC:
            return lambda x: 1 / (1 + np.exp(-x))
        elif self.name == ActivationFunction.TANH:
            return lambda x: np.tanh(x)
        elif self.name == ActivationFunction.RELU:
            return lambda x: np.maximum(0, x)
        elif self.name == ActivationFunction.STEP:
            return lambda x: 1 if x > 0 else -1
        elif self.name == ActivationFunction.IDENTITY:
            return lambda x: x
        # elif self.name == ActivationFunction.SOFTMAX:
        #     return lambda x: np.exp(x) / np.sum(np.exp(x))
        else:
            raise Exception("Unknown activation function")

    def __get_df_function__(self):
        if self.name == ActivationFunction.LOGISTIC:
            return lambda x: self.function(x) * (1 - self.function(x))
        elif self.name == ActivationFunction.TANH:
            return lambda x: 1 - self.function(x) ** 2
        elif self.name == ActivationFunction.RELU:
            return lambda x: 1 if x > 0 else 0
        elif self.name == ActivationFunction.STEP:
            return lambda x: 1
        elif self.name == ActivationFunction.IDENTITY:
            return lambda x: 1
        # elif self.name == ActivationFunction.SOFTMAX:
            # define jacobi matrix
            # @link https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1

            # return lambda x: x * (1 - x)
        else:
            raise Exception("Unknown activation function")

    def __call__(self, x):
        return self.function(x)

    def derivative(self, x):
        return self._derivative(x)
