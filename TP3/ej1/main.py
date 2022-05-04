import argparse
import numpy as np
from ..neural_network import NeuralNetwork


def get_dataset(operation="and"):
    if operation == "and":
        input_dataset = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
        expected_output = np.array([[-1], [-1], [-1], [1]])
    elif operation == "xor":
        input_dataset = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
        expected_output = np.array([[1], [1], [-1], [-1]])
    return input_dataset, expected_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="and", help="Datasets to choose from for the network to learn. Options are 'and' or 'xor'.", dest="dataset", required=False)

    args = parser.parse_args()

    input_dataset, expected_output = get_dataset(args.dataset)
    neural_network = NeuralNetwork(hidden_sizes=[], input_size=2, prediction_output_path=1, learning_rate=0.8,
                                   bias=0.5, activation_function_str="step", batch_size=1, prediction_output_path='output_layer_weights.txt')
    neural_network.train(input_dataset, expected_output, epochs=1000, tol=1e-8)

    error, predictions = neural_network.test(input_dataset, expected_output)
    print("Error: ", error)
    print("Predictions: ", predictions)
