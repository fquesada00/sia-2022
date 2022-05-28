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

    def __get_distances(self, input_vector):
        distances = []
        for i in range(self.k):
            for j in range(self.k):
                distance = np.linalg.norm(
                    input_vector-self.weights[i, j, :], ord=2)
                distances.append(distance)
        return distances

    def __get_winner(self, input_vector):

        distances = self.__get_distances(input_vector)
        winner_index_i = np.argmin(distances) // self.k
        winner_index_j = np.argmin(distances) % self.k
        winner = self.weights[winner_index_i, winner_index_j, :]
        winner_distance = distances[np.argmin(distances)]

        return winner_index_i, winner_index_j, winner, winner_distance

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

        u_matrix_arr = []
        
        time_constant = epochs / np.log(self.initial_r)
        for i in range(len(dataset_input[0])*(epochs + 1)):

            u_matrix_arr.append(self.get_u_matrix())

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

        return winner_idx_arr_row, winner_idx_arr_col, radius_arr, learning_rate_arr, dist_arr,u_matrix_arr

    def test(self, dataset_input):
        winners_sequence = []
        winners = []
        for input_value in dataset_input:
            winner_index_i, winner_index_j, winner, dist = self.__get_winner(
                input_value)
            winners_sequence.append((winner_index_i, winner_index_j))
            winners.append((winner_index_i, winner_index_j))

        return winners_sequence, winners

    def get_mean_column_weight(self, column):
        row_weights = self.weights[:, :, column].reshape(self.k*self.k)
        mean = np.mean(row_weights, axis=0)
        return row_weights.reshape(self.k, self.k) / mean
    
    def get_u_matrix(self):
        u_matrix = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                v = self.weights[i][j]  # a vector 
                sum_dists = 0.0; ct = 0
                if i-1 >= 0:    # above
                    sum_dists += np.linalg.norm(v - self.weights[i-1][j]); ct += 1
                if i+1 <= (self.k-1):   # below
                    sum_dists += np.linalg.norm(v - self.weights[i+1][j]); ct += 1
                if j-1 >= 0:   # left
                    sum_dists += np.linalg.norm(v - self.weights[i][j-1]); ct += 1
                if j+1 <= (self.k-1):   # right
                    sum_dists += np.linalg.norm(v - self.weights[i][j+1]); ct += 1
                u_matrix[i][j] = sum_dists / ct

        return u_matrix
