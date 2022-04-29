import numpy as np
from neural_network import NeuralNetwork

if __name__ == "__main__":
    dataset = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    expected_output = np.array([[-1], [-1], [-1], [1]])
    neural_network = NeuralNetwork(hidden_sizes=[], input_size=2, output_size=1, learning_rate=0.01,
                                   bias=0.5, activation_function="step", batch_size=1, output_file_name='mbeh.txt')
    neural_network.train(dataset, expected_output, epochs=10000, error=1e-8)

    neural_network.test(dataset, expected_output)
