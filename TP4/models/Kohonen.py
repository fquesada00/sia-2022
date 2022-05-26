import numpy as np
from pandas import array


class Kohonen():

    def __init__(self, k, dataset_input, initial_r, initial_lr):
        self.k = k
        self.initial_r = initial_r
        self.initial_lr = initial_lr
        self.__create_network(k, dataset_input)

    def __create_network(self, k, dataset_input: np.ndarray, ):

        self.weights = np.array([])

        for i in range(k):
            for j in range(k):
                random_input_index = np.random.choice(
                    dataset_input.shape[0], size=1)[0]

                self.weights = np.append(
                    self.weights, dataset_input[random_input_index, :])

        self.weights = self.weights.reshape(k, k, dataset_input.shape[1])

    def __get_winner(self, input_vector):
        winner_index_i, winner_index_j = 0, 0
        winner_distance = float('inf')
        o = 0
        # print("winner distance", winner_distance)
        for i in range(self.k):
            for j in range(self.k):
                distance = np.linalg.norm(
                    input_vector-self.weights[i, j, :], ord=2)
                if distance < winner_distance:
                    winner_distance = distance
                    winner_index_i = i
                    winner_index_j = j
        return winner_index_i, winner_index_j, self.weights[winner_index_i, winner_index_j, :], winner_distance

    def decay_radius(self, time, time_constant):
        return self.initial_r * np.exp(-time/time_constant)

    def decay_learning_rate(self, time, epochs):
        return self.initial_lr * np.exp(-time/epochs)

    def train(self, dataset_input, epochs):

        winner_idx_arr_row = []
        winner_idx_arr_col = []
        radius_arr = []
        learning_rate_arr = []
        dist_arr = []

        time_constant = epochs / np.log(self.initial_r)

        for i in range(epochs + 1):

            shuffled_inputs = np.random.permutation(dataset_input)

            for input_vector in shuffled_inputs:
                # Get the winner index
                winner_index_i, winner_index_j, winner, dist = self.__get_winner(
                    input_vector)

                winner_idx_arr_row.append(winner_index_i)
                winner_idx_arr_col.append(winner_index_j)
                dist_arr.append(dist)

                # Decay radius and learning rate

                radius = self.decay_radius(i, time_constant)
                lr = self.decay_learning_rate(i, epochs)

                radius_arr.append(radius)
                learning_rate_arr.append(lr)

                # Update the winner neighborhood
                for x in range(self.k):
                    for y in range(self.k):
                        w = self.weights[x, y, :]

                        neuron_distance = np.linalg.norm(
                            np.array([winner_index_i-x, winner_index_j-y]))

                        if neuron_distance <= radius:
                            self.weights[x, y, :] = w + lr*(input_vector-w)

        return winner_idx_arr_row, winner_idx_arr_col, radius_arr, learning_rate_arr, dist_arr

    def test(self, dataset_input):
        neurons = np.zeros(shape=(self.k, self.k))
        for input_value in dataset_input:
            winner_index_i, winner_index_j, winner, dist = self.__get_winner(
                input_value)
            neurons[winner_index_i, winner_index_j] += 1
        return neurons
