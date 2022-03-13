from enum import Enum


class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __str__(self):
        if self == Action.UP:
            return 'UP'

        if self == Action.RIGHT:
            return 'RIGHT'

        if self == Action.DOWN:
            return 'DOWN'

        if self == Action.LEFT:
            return 'LEFT'