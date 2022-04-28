import numpy as np
from neural_network import NeuralNetwork

if __name__ == "__main__":
    dataset = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    expected_output = np.array([[-1], [-1], [-1], [1]])
    neural_network = NeuralNetwork(hidden_sizes=[], input_size=2, output_size=1, learning_rate=0.1, bias=0.5, activation_function="step", batch_size=1)
    neural_network.train(dataset, expected_output)

