from enum import Enum
from search_methods.heuristics.admissible.euclidean import euclidean_heuristic
from search_methods.heuristics.admissible.manhattan import manhattan_heuristic
from search_methods.heuristics.admissible.misplaced_tiles import misplaced_tiles_heuristic
from search_methods.heuristics.inadmissible.misplaced_tiles_value import misplaced_tiles_value_heuristic
from search_methods.heuristics.inadmissible.visited_tiles_value import visited_tiles_value_heuristic


class Heuristic(Enum):
    MANHATTAN_DISTANCE = manhattan_heuristic
    EUCLIDEAN_DISTANCE = euclidean_heuristic
    MISPLACED_TILES = misplaced_tiles_heuristic
    VISITED_TILES_VALUE = visited_tiles_value_heuristic
    MISPLACED_TILES_VALUE = misplaced_tiles_value_heuristic

    def __str__(self):
        if self.name == 'MANHATTAN_DISTANCE':
            return 'Manhattan distance'
        elif self.name == 'EUCLIDEAN_DISTANCE':
            return 'Euclidean distance'
        elif self.name == 'MISPLACED_TILES':
            return 'Misplaced tiles'
        elif self.name == 'VISITED_TILES_VALUE':
            return 'Visited tiles value'
        elif self.name == 'MISPLACED_TILES_VALUE':
            return 'Misplaced tiles value'