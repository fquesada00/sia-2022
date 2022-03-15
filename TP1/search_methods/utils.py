import math
from search_methods.Action import Action
from search_methods.constants import BLANK, GOAL_STATE


def count_inversions(state, n):
    count = 0

    for i in range(0, n*n-1):
        for j in range(i+1, n*n):
            if state[j] != BLANK and state[i] != BLANK and state[i] > state[j]:
                count += 1

    return count

def get_tile_col(tile_index, n):
    return tile_index % n

def get_tile_row(tile_index, n):
    return math.floor(tile_index / n)

def is_solvable(initial_state, n):
    inversions = count_inversions(initial_state, n)

    if n % 2 == 1:
        return inversions % 2 == 0
    else:
        blank_index = find_blank_index(initial_state)
        row_from_bottom = n - get_tile_row(blank_index, n)

        if row_from_bottom % 2 == 0:
            return inversions % 2 == 1
        else:
            return inversions % 2 == 0


def get_next_state(state, action, n):
    new_state = state.copy()
    blank_index = find_blank_index(state)

    if action == Action.UP:
        new_state[blank_index], new_state[blank_index -
                                          n] = new_state[blank_index-n], new_state[blank_index]
    elif action == Action.RIGHT:
        new_state[blank_index], new_state[blank_index +
                                          1] = new_state[blank_index+1], new_state[blank_index]
    elif action == Action.DOWN:
        new_state[blank_index], new_state[blank_index +
                                          n] = new_state[blank_index+n], new_state[blank_index]
    elif action == Action.LEFT:
        new_state[blank_index], new_state[blank_index -
                                          1] = new_state[blank_index-1], new_state[blank_index]

    return new_state


def get_actions(state, n):
    blank_index = find_blank_index(state)
    blank_col = blank_index % n
    actions = []

    if blank_col > 0:
        actions.append(Action.LEFT)
    if blank_index <= n*(n - 1) - 1:
        actions.append(Action.DOWN)
    if blank_col < n-1:
        actions.append(Action.RIGHT)
    if blank_index >= n:
        actions.append(Action.UP)

    return actions


def is_goal_state(state):
    return state == GOAL_STATE


def is_blank(tile):
    return tile == BLANK


def find_blank_index(state):
    return state.index(BLANK)