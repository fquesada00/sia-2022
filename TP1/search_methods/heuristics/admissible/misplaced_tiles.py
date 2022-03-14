from search_methods.constants import BLANK
from search_methods.heuristics.utils import sum_over_tiles_heuristic


def is_tile_in_correct_position(tile_index, tile, n): 
    if tile_index == tile - 1 or tile == BLANK:
        return 0
    return 1

def misplaced_tiles_heuristic(state, n):
    return sum_over_tiles_heuristic(state, is_tile_in_correct_position, n)