from constants import FITNESS_DISTANCE, MAX_TIME, MIN_FITNESS, REQUIRED_GENERATION_REPEATS
from constants import MAX_GENERATIONS
from .max_generations_cut_condition import max_generations_cut_condition
from .fitness_value_cut_condition import fitness_value_cut_condition
from .fitness_variation_cut_condition import fitness_variation_cut_condition
from .max_time_cut_condition import max_time_cut_condition
from .max_generations_cut_condition import max_generations_cut_condition

from enum import Enum


class CutCondition(Enum):
    MAX_GENERATIONS = 0
    FITNESS_VALUE = 1
    FITNESS_VARIATION = 2
    MAX_TIME = 3


class CutConditionParameters():

    def __init__(self, cut_condition_method=CutCondition.MAX_GENERATIONS, max_generations=MAX_GENERATIONS, min_fitness_value=MIN_FITNESS, fitness_distance=FITNESS_DISTANCE, fitness_required_generations_repeats=REQUIRED_GENERATION_REPEATS, max_time=MAX_TIME):
        self._cut_condition_method = cut_condition_method
        self._max_generations = max_generations
        self._min_fitness_value = min_fitness_value
        self._fitness_distance = fitness_distance
        self._fitness_required_generations_repeats = fitness_required_generations_repeats
        self._max_time = max_time

    @property
    def max_generations(self):
        return self._max_generations

    @max_generations.setter
    def max_generations(self, value):
        self._max_generations = value

    @property
    def min_fitness_value(self):
        return self._min_fitness_value

    @min_fitness_value.setter
    def min_fitness_value(self, value):
        self._min_fitness_value = value

    @property
    def fitness_distance(self):
        return self._fitness_distance

    @fitness_distance.setter
    def fitness_distance(self, value):
        self._fitness_distance = value

    @property
    def fitness_required_generations_repeats(self):
        return self._fitness_required_generations_repeats

    @fitness_required_generations_repeats.setter
    def fitness_required_generations_repeats(self, value):
        self._fitness_required_generations_repeats = value

    @property
    def max_time(self):
        return self._max_time

    @max_time.setter
    def max_time(self, value):
        self._max_time = value

    @property
    def cut_condition_method(self):
        if self._cut_condition_method == CutCondition.MAX_GENERATIONS:
            return max_generations_cut_condition
        elif self._cut_condition_method == CutCondition.FITNESS_VALUE:
            return fitness_value_cut_condition
        elif self._cut_condition_method == CutCondition.FITNESS_VARIATION:
            return fitness_variation_cut_condition
        elif self._cut_condition_method == CutCondition.MAX_TIME:
            return max_time_cut_condition
        else:
            raise Exception("Cut condition method not found")


    def __str__(self):
        return "CutConditionMethod: {}, MaxGenerations: {}, MinFitnessValue: {}, FitnessDistance: {}, FitnessRequiredGenerationsRepeats: {}, MaxTime: {}".format(self._cut_condition_method, self._max_generations, self._min_fitness_value, self._fitness_distance, self._fitness_required_generations_repeats, self._max_time)
