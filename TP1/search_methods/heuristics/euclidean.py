import math
from search_methods.constants import BLANK
from search_methods.heuristics.utils import  sum_over_tiles_heuristic
from search_methods.utils import get_tile_col, get_tile_row


def euclidean_distance(tile_index, tile, n):
    if tile_index == tile - 1 or tile == BLANK:
        return 0

    tile_row = get_tile_row(tile_index, n)
    tile_col = get_tile_col(tile_index, n)

    tile_row_goal_position = get_tile_row(tile - 1, n)
    tile_col_goal_position = get_tile_col(tile - 1, n);

    row_distance = abs(tile_row - tile_row_goal_position)
    col_distance = abs(tile_col - tile_col_goal_position)

    distance = math.sqrt(row_distance*row_distance + col_distance*col_distance)
    return distance

def euclidean_heuristic(state, n):
  return sum_over_tiles_heuristic(state, euclidean_distance, n)