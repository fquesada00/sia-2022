from ...constants import MULTIPLE_POINT_CROSSOVER_POINTS;
from .multiple_point_crossover import multiple_point_crossover
from .single_point_crossover import single_point_crossover
from .uniform_crossover import uniform_crossover

from enum import Enum

class CrossoverMethod(Enum):
    MULTIPLE_POINT = 0
    SINGLE_POINT = 1
    UNIFORM = 2

    @staticmethod
    def from_str(label):
        if label in ("multiple_point", "MULTIPLE_POINT"):
            return CrossoverMethod.MULTIPLE_POINT
        elif label in ("single_point", "SINGLE_POINT"):
            return CrossoverMethod.SINGLE_POINT
        elif label in ("uniform", "UNIFORM"):
            return CrossoverMethod.UNIFORM
        else:
            raise ValueError(label+ " has no value matching")

    def __str__(self):
        if self == CrossoverMethod.MULTIPLE_POINT:
            return "Multiple point"
        elif self == CrossoverMethod.SINGLE_POINT:
            return "Single point"
        elif self == CrossoverMethod.UNIFORM:
            return "Uniform"
        else:
            return "Unknown"

class CrossoverParameters():

    def __init__(self,crossover_method=CrossoverMethod.SINGLE_POINT,multiple_point_crossover_points=MULTIPLE_POINT_CROSSOVER_POINTS):
        self._crossover_method = crossover_method
        self._multiple_point_crossover_points = multiple_point_crossover_points

    @property
    def multiple_point_crossover_points(self):
        return self._multiple_point_crossover_points

    @multiple_point_crossover_points.setter
    def multiple_point_crossover_points(self, value):
        self._multiple_point_crossover_points = value
        
    @property
    def crossover_method_name(self):
        return self._crossover_method

    @property
    def crossover_method(self):
        if self._crossover_method == CrossoverMethod.MULTIPLE_POINT:
            return multiple_point_crossover
        elif self._crossover_method == CrossoverMethod.SINGLE_POINT:
            return single_point_crossover
        elif self._crossover_method == CrossoverMethod.UNIFORM:
            return uniform_crossover
        else:
            return None

    @crossover_method.setter
    def crossover_method(self,value):
        self._crossover_method = value

    def __str__(self):
        return "Crossover method {}, MultiplePointCrossoverPoints {}".format(self._crossover_method,self._multiple_point_crossover_points) 