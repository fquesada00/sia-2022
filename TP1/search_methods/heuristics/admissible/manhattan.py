
from search_methods.constants import BLANK
from search_methods.heuristics.utils import  sum_over_tiles_heuristic
from search_methods.utils import get_tile_col, get_tile_row


def manhattan_distance(tile_index, tile, n):
    if tile_index == tile - 1 or tile == BLANK:
        return 0

    tile_row = get_tile_row(tile_index, n)
    tile_col = get_tile_col(tile_index, n)

    tile_row_goal_position = get_tile_row(tile - 1, n)
    tile_col_goal_position = get_tile_col(tile - 1, n);

    distance = abs(tile_row - tile_row_goal_position) + abs(tile_col - tile_col_goal_position)
    return distance

def manhattan_heuristic(state, n):
    return sum_over_tiles_heuristic(state, manhattan_distance, n)