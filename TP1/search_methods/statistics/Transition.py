class Transition:
    def __init__(self, parent_action, state):
        self.parent_action = str(parent_action)
        self.state = state
    def __str__(self) -> str:
        return f'{self.parent_action} -> {self.state}'
