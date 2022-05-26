import numpy as np


class Hopfield():

    def __init__(self, inputs: list(list(list(int)))):
        # input is [[first_pattern], [second_pattern], ..., [last_pattern]]
        self.number_of_neurons = len(np.ndarray.flatten(inputs[0]))
        self.number_of_inputs = len(inputs)

        flattened_inputs = []
        for _input in inputs:
            flattened_inputs.append(np.ndarray.flatten(_input))

        self.patterns = flattened_inputs
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        # calculate matrix product of input and its transpose
        return (1 / self.number_of_neurons) * np.dot(self.patterns.T, self.patterns) - np.eye(self.number_of_neurons)

    def print_state(self, state: list(int)):
        for i in range(len(state)):
            if i % 6 == 1:
                print()
            print(f"{'*' if state[i] == 1 else ' '}")


    def matches(self, pattern: list(int)):
        # check if the pattern matches any of the patterns in the input
        for i in range(self.number_of_inputs):
            if np.array_equal(pattern, self.inputs[i]):
                return True
        
        return False

    def associate(self, pattern: list(list(int))):
        # calculate the output of the network
        state = np.ndarray.flatten(pattern)
        has_converged = False
        
        self.print_state(sign)

        while not has_converged:
            sign = np.sign(np.dot(self.weights, state))
            self.print_state(sign)
            state = sign if np.count_nonzero(sign) < self.number_of_neurons else state
            has_converged = self.matches(state)
        
        return state





