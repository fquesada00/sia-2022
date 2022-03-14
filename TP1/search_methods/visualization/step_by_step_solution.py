import sys
from domonic.html import *
from search_methods.Action import Action
import search_methods.constants as constants


class Transition:
    def __init__(self, action, state):
        self.action = action
        self.state = state


def get_direction_emoji(action):
    if action == Action.UP:
        return '⬆️'
    if action == Action.DOWN:
        return '⬇️'
    if action == Action.LEFT:
        return '⬅️'
    if action == Action.RIGHT:
        return '➡️'
    else:
        return ''


def generate_html_output(goal_node, file_name):
    html_steps = []
    current = goal_node
    step_number = goal_node.path_cost

    while current is not None:
        tiles = list(map(lambda tile: div(
            '' if tile == constants.BLANK else tile, _class='number-tile'), current.state))
        html_steps.insert(0, div(b(f'Step {step_number}:'), p('Begin' if current.action is None else f'Move blank {str(current.action).lower()}'),
                                 p(get_direction_emoji(current.action),
                                   _class='arrow-emoji'), div(*tiles, _class='board'),
                                 _class=f'{ "step" if step_number != goal_node.path_cost else "final-step"}'))
        step_number -= 1
        current = current.parent

    original_stdout = sys.stdout

    with open(file_name, 'w') as f:
        sys.stdout = f
        dom = html(head(link(_href="styles.css", _rel="stylesheet")),
                   body(div(*html_steps, _class='steps-container')))
        print(f'{dom}')
        sys.stdout = original_stdout
