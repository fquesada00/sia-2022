from enum import Enum
from search_methods.heuristics.euclidean import euclidean_heuristic
from search_methods.heuristics.manhattan import manhattan_heuristic
from search_methods.heuristics.misplaced_tiles import misplaced_tiles_heuristic
from search_methods.heuristics.inadmissible.misplaced_tiles_value import misplaced_tiles_value_heuristic
from search_methods.heuristics.inadmissible.visited_tiles_value import visited_tiles_value_heuristic


class Heuristic(Enum):
    MANHATTAN_DISTANCE = manhattan_heuristic
    EUCLIDEAN_DISTANCE = euclidean_heuristic
    MISPLACED_TILES = misplaced_tiles_heuristic
    VISITED_TILES_VALUE = visited_tiles_value_heuristic
    MISPLACED_TILES_VALUE = misplaced_tiles_value_heuristic