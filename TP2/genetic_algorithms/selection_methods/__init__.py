from constants import INITIAL_TEMPERATURE,FINAL_TEMPERATURE,EXP_RATE,K,TOURNAMENT_THRESHOLD
from .uniform_selection import uniform_selection
from .roulette_selection import roulette_selection
from .tournament_selection import tournament_selection
from .elite_selection import elite_selection
from .boltzmann_selection import boltzmann_selection
from .rank_selection import rank_selection
from .truncate_selection import truncate_selection

from enum import Enum

class SelectionMethod(Enum):
    UNIFORM = 0
    ROULETTE = 1
    TOURNAMENT = 2
    ELITE = 3
    BOLTZMANN = 4
    RANK = 5
    TRUNCATE = 6

    @staticmethod
    def from_str(label):
        if label in ("uniform", "UNIFORM"):
            return SelectionMethod.UNIFORM
        elif label in ("roulette", "ROULETTE"):
            return SelectionMethod.ROULETTE
        elif label in ("tournament", "TOURNAMENT"):
            return SelectionMethod.TOURNAMENT
        elif label in ("elite", "ELITE"):
            return SelectionMethod.ELITE
        elif label in ("boltzmann", "BOLTZMANN"):
            return SelectionMethod.BOLTZMANN
        elif label in ("rank", "RANK"):
            return SelectionMethod.RANK
        elif label in ("truncate", "TRUNCATE"):
            return SelectionMethod.TRUNCATE        
        else:
            raise ValueError(label+ " has no value matching")

    def __str__(self):
        if self == SelectionMethod.UNIFORM:
            return "Uniform"
        elif self == SelectionMethod.ROULETTE:
            return "Roulette"
        elif self == SelectionMethod.TOURNAMENT:
            return "Tournament"
        elif self == SelectionMethod.ELITE:
            return "Elite"
        elif self == SelectionMethod.BOLTZMANN:
            return "Boltzmann"
        elif self == SelectionMethod.RANK:
            return "Rank"
        elif self == SelectionMethod.TRUNCATE:
            return "Truncate"

class SelectionParameters():

    def __init__(self, selection_method = SelectionMethod.ROULETTE, initial_temperature=INITIAL_TEMPERATURE, final_temperature=FINAL_TEMPERATURE,exp_rate = EXP_RATE,k=K,threshold=TOURNAMENT_THRESHOLD):
        self._selection_method = selection_method
        self._initial_temperature = initial_temperature
        self._final_temperature = final_temperature
        self._exp_rate = exp_rate
        self._k = k
        self._threshold = threshold


    @property
    def initial_temperature(self):
        return self._initial_temperature
    
    @initial_temperature.setter
    def initial_temperature(self, value):
        self._initial_temperature = value

    @property
    def final_temperature(self):
        return self._final_temperature
    
    @final_temperature.setter
    def final_temperature(self, value):
        self._final_temperature = value

    @property
    def exp_rate(self):
        return self._exp_rate
        
    @exp_rate.setter
    def exp_rate(self, value):
        self._exp_rate = value
    
    @property
    def k(self):
        return self._k
    
    @k.setter
    def k(self, value):
        self._k = value

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self,value):
        self._threshold = value
    
    @property
    def selection_method_name(self):        
        return self._selection_method


    @property
    def selection_method(self):
        if self._selection_method == SelectionMethod.UNIFORM:
            return uniform_selection
        elif self._selection_method == SelectionMethod.ROULETTE:
            return roulette_selection
        elif self._selection_method == SelectionMethod.TOURNAMENT:
            return tournament_selection
        elif self._selection_method == SelectionMethod.ELITE:
            return elite_selection
        elif self._selection_method == SelectionMethod.BOLTZMANN:
            return boltzmann_selection
        elif self._selection_method == SelectionMethod.RANK:
            return rank_selection
        elif self._selection_method == SelectionMethod.TRUNCATE:
            return truncate_selection
        else:
            raise ValueError("Invalid selection method")

    @selection_method.setter
    def selection_method(self, value):
        self._selection_method = value

    def __str__(self):
        return "Selection method: {}, Initial temperature: {}, Final temperature: {}, Exp rate: {}, K: {}".format(self._selection_method, self._initial_temperature, self._final_temperature, self._exp_rate, self._k)
