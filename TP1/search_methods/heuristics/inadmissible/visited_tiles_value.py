from search_methods.Action import Action
from search_methods.constants import BLANK
from search_methods.heuristics.utils import sum_over_tiles_heuristic
from search_methods.utils import get_tile_col, get_tile_row
from functools import reduce

def sum_over_range_with_lambda(reducer, range):
    return reduce(lambda x, y: x + y, list(map(reducer, range)))

def execute_actions(actions, tile_row_position, tile_col_position, tile_row_goal_position, tile_col_goal_position, n):
    sum = 0

    for action in actions:
        if action == Action.UP:
            sum += sum_over_range_with_lambda(lambda x: x * n + 1, range(tile_row_position - tile_row_goal_position + 1))
        elif action == Action.DOWN:
            sum += sum_over_range_with_lambda(lambda x: x * (n + 1) + 1, range(tile_row_goal_position - tile_row_position + 1))
        elif action == Action.RIGHT:
            sum += sum_over_range_with_lambda(lambda x: x + tile_col_position + 1 + n * tile_row_position, range(tile_col_goal_position - tile_col_position + 1))
        elif action == Action.LEFT:
            sum += sum_over_range_with_lambda(lambda x: x + tile_col_goal_position + 1 + n * tile_row_position, range(tile_col_position - tile_col_goal_position + 1))
    
    return sum
        
def max_sum_tiles_to_visit(tile_index, tile, n): 
    if tile_index == tile - 1 or tile == BLANK:
        return 0
    
    tile_row = get_tile_row(tile_index, n)
    tile_col = get_tile_col(tile_index, n)

    tile_row_goal_position = get_tile_row(tile - 1, n)
    tile_col_goal_position = get_tile_col(tile - 1, n);

    row_distance = tile_row_goal_position - tile_row
    col_distance = tile_col_goal_position - tile_col

    actions = []

    # one movement tile => the sum is the goal tile value
    if (abs(row_distance) == 1 and col_distance == 0) or (abs(col_distance) == 1 and row_distance == 0):
        return tile_row_goal_position * n + tile_col_goal_position + 1

    # worst case is [DOWN, RIGHT, LEFT, UP]

    # go down
    if row_distance > 0:
        actions.append(Action.DOWN)
        # go right
        if col_distance > 0:
            actions.append(Action.RIGHT)
        # go left
        elif col_distance < 0:
            actions.append(Action.LEFT)
    else:
        # go right
        if col_distance > 0:
            actions.append(Action.RIGHT)
        # go left
        elif col_distance < 0:
            actions.append(Action.LEFT)

        # go up
        if row_distance > 0:
            actions.append(Action.UP)

    
    return execute_actions(actions, tile_row, tile_col, tile_row_goal_position, tile_col_goal_position, n)

def visited_tiles_value_heuristic(state, n):
    return sum_over_tiles_heuristic(state, max_sum_tiles_to_visit, n)
