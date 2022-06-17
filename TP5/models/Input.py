from .Layer import Layer


class Input(Layer):
    def __init__(self, shape=None, name=None):
        super().__init__(shape[0])
        self.shape = shape
        self.name = name
