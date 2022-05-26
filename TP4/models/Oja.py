import numpy as np


class Oja:
    @classmethod
    def compute_pc1(input_dataset: np.ndarray, epochs: int, learning_rate: float):
        """
        Approximate the first principal component of the input dataset.

        :param input_dataset: The input dataset.
        :param epochs: The number of epochs.
        :return: The first principal component approximation.
        """
        
        # Initialize the weights with random values and norm 1.
        weights = np.random.rand(input_dataset.shape[1])
        weights /= np.linalg.norm(weights)


        # For each epoch.
        for epoch in range(epochs):
            # For each sample.
            for sample in input_dataset:
                # Compute the projection of the sample on the weights.
                proj = np.dot(sample, weights)
                # Update the weights.
                weights += learning_rate * proj * (sample - proj * weights)

        return weights        