import numpy as np


class Kohonen():

    def __create_network(self, k, dataset_input: np.ndarray, ):
        self.weights = np.array([])
        for i in range(k):
            for j in range(k):
                random_input_index = np.random.choice(
                    dataset_input.shape[0], size=1)
                self.weights = np.append(
                    self.weights, dataset_input[random_input_index, :])
        self.weights = self.weights.reshape(k, k, dataset_input.shape[1])

    def __init__(self, k, dataset_input,initial_r, initial_lr):
        self.k = k
        self.initial_r = initial_r
        self.initial_lr = initial_lr
        self.__create_network(k, dataset_input)

    def __get_winner(self, input_vector):
        winner_index_i, winner_index_j = 0, 0
        winner_distance = np.linalg.norm(input_vector-self.weights[0, 0, :])
        for i in range(self.k):
            for j in range(self.k):
                distance = np.linalg.norm(input_vector-self.weights[i, j, :])
                if distance < winner_distance:
                    winner_distance = distance
                    winner_index_i = i
                    winner_index_j = j
        return winner_index_i, winner_index_j, self.weights[winner_index_i, winner_index_j, :], winner_distance


    def decay_radius(self, time,time_constant):
        return self.initial_r * np.exp(-time/time_constant)

    def decay_learning_rate(self, time,n_iterations):
        return self.initial_lr * np.exp(-time/n_iterations)

    def train(self, dataset_input, n_iterations):

        winner_idx_arr_row = []
        winner_idx_arr_col = []
        radius_arr = []
        learning_rate_arr = []
        dist_arr = []

        time_constant = n_iterations / np.log(self.initial_r)

        for i in range(n_iterations):

            # Choose a random element from the input values
            random_input_index = np.random.choice(
                dataset_input.shape[0], size=1)
            input_vector = dataset_input[random_input_index, :]

            # Get the winner index
            winner_index_i, winner_index_j,winner,dist = self.__get_winner(
                input_vector)

            winner_idx_arr_row.append(winner_index_i)
            winner_idx_arr_col.append(winner_index_j)
            dist_arr.append(dist)

            # Decay radius and learning rate

            radius = self.decay_radius(i, time_constant)
            lr = self.decay_learning_rate(i, n_iterations)

            radius_arr.append(radius)
            learning_rate_arr.append(lr)

            # Update the winner neighborhood
            for x in range(self.k):
              for y in range(self.k):
                w = self.weights[x, y, :]

                w_dist = np.linalg.norm(w-winner)

                if w_dist <= radius:
                    self.weights[x, y, :] = w + lr*(input_vector-w)

        return winner_idx_arr_row, winner_idx_arr_col, radius_arr, learning_rate_arr, dist_arr

    def test(self,dataset_input):
      neurons = np.zeros(shape=(self.k,self.k))
      for input_value in dataset_input:
        winner_index_i, winner_index_j,winner,dist = self.__get_winner(input_value)
        neurons[winner_index_i,winner_index_j] += 1
      return neurons

