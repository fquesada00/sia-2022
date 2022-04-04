from constants import MULTIPLE_POINT_CROSSOVER_POINTS;
from .multiple_point_crossover import multiple_point_crossover
from .single_point_crossover import single_point_crossover
from .uniform_crossover import uniform_crossover

from enum import Enum

class CrossoverMethod(Enum):
    MULTIPLE_POINT = 0
    SINGLE_POINT = 1
    UNIFORM = 2


class CrossoverParameters():

    def __init__(self,crossover_method=CrossoverMethod.SINGLE_POINT,multiple_point_crossover_points=MULTIPLE_POINT_CROSSOVER_POINTS,):
        self._crossover_method = crossover_method
        self._multiple_point_crossover_points = multiple_point_crossover_points

    @property
    def multiple_point_crossover_points(self):
        return self._multiple_point_crossover_points
        

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

    def __str__(self):
        return "CrossoverMethod {}, MultiplePointCrossoverPoints {}".format(self._crossover_method,self._multiple_point_crossover_points) 