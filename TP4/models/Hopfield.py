import numpy as np


class Hopfield:

    def __init__(self, inputs: list[list[list[int]]]):
        # input is [[first_pattern], [second_pattern], ..., [last_pattern]]
        self.number_of_neurons = len(np.array(inputs[0]).flatten())

        flattened_inputs = []
        for _input in inputs:
            flattened_inputs.append(np.array(_input).flatten())

        self.patterns = np.array(flattened_inputs)

        self.weights = self.initialize_weights()

        print(np.diag(self.weights))

    def initialize_weights(self):
        # calculate matrix product of input and its transpose
        matrix = (1 / self.number_of_neurons) * np.dot(self.patterns.T, self.patterns)

        # set diagonal to zero
        np.fill_diagonal(matrix, 0)

        return matrix

    def print_state(self, state: list[int]):
        for i in range(len(state)):
            print(f"{'*' if state[i] == 1 else ' '}", end="")
            if i % 5 == 4:
                print()
        print()

    def energy(self, state: list[int]):
        H = 0

        for index, weight_row in enumerate(self.weights):
            H += np.dot(weight_row, state) * state[index]
        
        return H

    def associate(self, pattern: list[list[int]]):
        # calculate the output of the network
        prev_state = np.array(pattern).flatten()
        has_converged = False

        self.print_state(prev_state)
        print(self.energy(prev_state))
        while not has_converged:
            sign = np.sign(np.dot(self.weights, prev_state))
            self.print_state(sign)
            state = sign if np.count_nonzero(sign) > 0 else prev_state
            energy = self.energy(state)
            print(energy)
            has_converged = np.array_equal(state, prev_state)
            break
        
        return state
