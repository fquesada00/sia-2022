import numpy as np


class Oja:

    # angle_and_error_vs_learning_rate_fig = plt.figure(2)

    @classmethod
    def compute_pc1(self, input_dataset: np.ndarray, epochs: int, learning_rate: float, generate_plot_data: bool = True):
        """
        Approximate the first principal component of the input dataset.

        :param input_dataset: The input dataset.
        :param epochs: The number of epochs.
        :param learning_rate: The learning rate.
        :param generate_plot_data: Whether to generate plot data.
        :return: The first principal component approximation.
        """

        # Initialize the weights with random values and norm 1.
        weights = np.random.uniform(
            size=input_dataset.shape[1], low=-1, high=1)
        weights /= np.linalg.norm(weights)

        # for plots
        if generate_plot_data:
            weights_evolution = [weights]
            iterations = [0]

        # For each epoch.
        for epoch in range(epochs):
            # For each sample.
            for i, sample in enumerate(input_dataset):
                # Compute the projection of the sample on the weights.
                proj = np.dot(sample, weights)
                # Update the weights.
                weights += learning_rate * proj * (sample - proj * weights)

                # for plots
                if generate_plot_data:
                    iterations.append(epoch * input_dataset.shape[0] + i)
                    weights_evolution.append(weights.copy())

        # for plots
        if generate_plot_data:
            return weights, weights_evolution, iterations

        return weights
