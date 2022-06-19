import numpy as np
import warnings
warnings.filterwarnings("error")


class ActivationFunction():
    LOGISTIC = "logistic"
    TANH = "tanh"
    RELU = "relu"
    STEP = "step"
    IDENTITY = "identity"
    # SOFTMAX = "softmax"

    def __init__(self, name, beta=1):
        self.name = name
        self.beta = beta
        self.function = self.__get_function__()

    def __str__(self):
        return self.name

    def __get_function__(self):
        if self.name == ActivationFunction.LOGISTIC:
            def logistic(x):
                try:
                    value = 1 / (1 + np.exp(-self.beta * x))
                except RuntimeWarning:
                    value = np.zeros(x.shape)
                return value
            return logistic
        elif self.name == ActivationFunction.TANH:
            return lambda x: np.tanh(self.beta * x)
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

    def __call__(self, x):
        return self.function(x)
