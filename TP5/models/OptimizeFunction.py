import numpy as np
from scipy.optimize import minimize
from autograd.misc.optimizers import adam
import numdifftools as nd


class OptimizerFunction():
    ADAM = "adam"
    POWELL = "powell"

    def __init__(self, name):
        self.name = name
        self.function = self.__get_function__()

    def __str__(self):
        return self.name

    def __get_function__(self):
        if self.name == OptimizerFunction.ADAM:
            return lambda loss_function, weights, num_iters=10, step_size=None: adam(nd.Gradient(loss_function), weights, num_iters=num_iters, step_size=step_size)
        elif self.name == OptimizerFunction.POWELL:
            return lambda loss_function, weights, num_iters=1000, step_size=None: minimize(loss_function, weights, method="Powell", options={'maxiter': num_iters, "disp": True})['x']
        else:
            raise Exception("Unknown optimizer function")

    def __call__(self, *args, **kwds):
        return self.function(*args, **kwds)
