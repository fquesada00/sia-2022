from enum import Enum
from search_methods.heuristics.euclidean import euclidean_heuristic
from search_methods.heuristics.manhattan import manhattan_heuristic
from search_methods.heuristics.misplaced_tiles import misplaced_tiles_heuristic


class Heuristic(Enum):
    MANHATTAN_DISTANCE = manhattan_heuristic
    EUCLIDEAN_DISTANCE = euclidean_heuristic
    MISPLACED_TILES = misplaced_tiles_heuristic