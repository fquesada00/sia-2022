from ...constants import MUTATION_RATE, NORMAL_MUTATION_STD, UNIFORM_MUTATION_BOUND
from .uniform_mutation import uniform_mutation
from .swap_mutation import swap_mutation
from .normal_mutation import normal_mutation

from enum import Enum


class MutationMethod(Enum):
    UNIFORM = 0
    SWAP = 1
    NORMAL = 2

    @staticmethod
    def from_str(label):
        if label in ("uniform", "UNIFORM"):
            return MutationMethod.UNIFORM
        elif label in ("swap", "SWAP"):
            return MutationMethod.SWAP
        elif label in ("normal", "NORMAL"):
            return MutationMethod.NORMAL
        else:
            raise ValueError(label+ " has no value matching")

    def __str__(self):
        if self == MutationMethod.UNIFORM:
            return "Uniform Mutation"
        elif self == MutationMethod.SWAP:
            return "Swap Mutation"
        elif self == MutationMethod.NORMAL:
            return "Normal Mutation"
        else:
            return "Unknown Mutation Method"

class MutationParameters():
    def __init__(self, mutation_method = MutationMethod.UNIFORM, mutation_rate = MUTATION_RATE, uniform_mutation_bound = UNIFORM_MUTATION_BOUND, normal_mutation_std = NORMAL_MUTATION_STD):
        self._mutation_method= mutation_method
        self._mutation_rate = mutation_rate
        self._uniform_mutation_bound = uniform_mutation_bound
        self._normal_mutation_std = normal_mutation_std

    @property
    def mutation_rate(self):
        return self._mutation_rate

    @mutation_rate.setter
    def mutation_rate(self, value):
        self._mutation_rate = value

    @property
    def uniform_mutation_bound(self):
        return self._uniform_mutation_bound

    @uniform_mutation_bound.setter
    def uniform_mutation_bound(self, value):
        self._uniform_mutation_bound = value

    @property
    def normal_mutation_std(self):
        return self._normal_mutation_std

    @normal_mutation_std.setter
    def normal_mutation_std(self, value):
        self._normal_mutation_std = value

    @property
    def mutation_method_name(self):
        return self._mutation_method

    @property
    def mutation_method(self):
        if self._mutation_method == MutationMethod.UNIFORM:
            return uniform_mutation
        elif self._mutation_method == MutationMethod.SWAP:
            return swap_mutation
        elif self._mutation_method == MutationMethod.NORMAL:
            return normal_mutation
        else:
            return None

    @mutation_method.setter
    def mutation_method(self, value):
        self._mutation_method = value

    def __str__(self):
        return "Mutation method: {}, MutationRate: {}, UniformMutationBound: {}, NormalMutationStd: {}".format(self._mutation_method, self._mutation_rate, self._uniform_mutation_bound, self._normal_mutation_std)
