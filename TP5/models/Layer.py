class Layer():
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim
