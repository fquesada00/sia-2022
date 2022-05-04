import argparse
import numpy as np

from ..metrics.confusion_matrix import generate_confusion_matrix
from ..metrics import print_metrics
from ..neural_network import NeuralNetwork
from ..metrics.holdout_validation import holdout_eval
from ..metrics.k_fold_cross_validation import k_fold_cross_validation_eval


def get_activation_function(activation_function):
    if activation_function == "non-linear":
        return "logistic"
    if activation_function == "linear":
        return "identity"


def get_dataset(dataset="xor", input_dataset_file_name=None, expected_output_dataset_file_name=None):
    input_dataset = expected_output = []

    if (dataset == "parity" or dataset == "numbers") and input_dataset_file_name is not None and expected_output_dataset_file_name is not None:
        input_dataset = parse(input_dataset_file_name)

        expected_output = parse_output_dataset(
            expected_output_dataset_file_name, 1 if dataset == "parity" else 10)

    elif dataset == "xor":
        input_dataset = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
        expected_output = np.array([[1], [1], [-1], [-1]])

    return input_dataset, expected_output


def parse_number_output_dataset(output_dataset_file_name, dataset_size):
    dataset = np.zeros((10, dataset_size), dtype=int)

    with open(output_dataset_file_name) as f:

        for i, line in enumerate(f):
            dataset[i] = np.array(line.split())

    return dataset


def parse_number_input_dataset(dataset_file_name):
    dataset = np.zeros((10, 5 * 7), dtype=int)

    with open(dataset_file_name, 'r') as dataset_file:
        data_index = 0
        data = []
        for i, line in enumerate(dataset_file):
            line = line.split()
            data.append(line)

            # when the matrix has been fully parsed, reshape it into an array
            if i % 7 == 6:
                data = np.array(data).reshape(5*7)
                dataset[data_index] = data
                data_index += 1
                data = []

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str,
                        default="xor", help="Dataset to use in the neural network.\n Value may be 'xor', 'parity' or 'numbers'", dest="dataset", required=False)
    parser.add_argument("--function", type=str,
                        default="linear", help="Activation function to use with perceptron.\n Value may be 'linear' or 'non-linear'", dest="activation_function", required=False)
    parser.add_argument("--split_method", help="Method to split input data into train and test set.\n Use 'all' to use the entire dataset for training, 'holdout' for holdout validation, 'k-fold' for k-fold cross validation.",
                        dest='training_method', required=False, default='all')
    parser.add_argument("--k", help="Number of folds to use for k-fold cross validation.",
                        dest='k', required=False, default=5)
    parser.add_argument("--ratio", help="Ratio of training data to use for holdout validation.",
                        dest='ratio', required=False, default=0.8)

    args = parser.parse_args()

    activation_function = get_activation_function(args.activation_function)

    input_dataset, expected_output = get_dataset(
        args.dataset, "TP3/ej3/datasets/input.txt", "TP3/ej3/datasets/numbers_expected_output.txt")

    # TODO: parametrize hidden sizes, input_size, output_size
    neural_network = NeuralNetwork(hidden_sizes=[3], input_size=2, prediction_output_path=1, learning_rate=0.05,
                                   bias=0.5, activation_function_str=activation_function, batch_size=1, beta=1)
    test_set_input = input_dataset
    test_set_expected_output = expected_output

    neural_network.train(input_dataset, expected_output,
                         epochs=10000, tol=0.000001)
    error, predictions = neural_network.test(
        test_set_input, test_set_expected_output)
    # expected_output = parse_dataset('./TP3/ej2/dataset/expected_output.txt')

    # training_method = args.training_method
    # k = int(args.k)
    # training_ratio = float(args.ratio)

    # if training_method == 'all':

    # elif training_method == 'holdout':
    #     error, holdout_sets = holdout_eval(input_dataset, expected_output, neural_network, activation_function=activation_function, training_ratio=training_ratio)

    #     test_set_input = holdout_sets['test_set'][0]
    #     test_set_expected_output = holdout_sets['test_set'][1]

    #     error, predictions = neural_network.test(test_set_input, test_set_expected_output)

    # elif training_method == 'k-fold':
    #     average_error, min_error_sets = k_fold_cross_validation_eval(input_dataset, expected_output, neural_network, k=k, activation_function=activation_function)

    #     test_set_input = min_error_sets['test_set'][0]
    #     test_set_expected_output = min_error_sets['test_set'][1]

    #     error, predictions = neural_network.test(test_set_input, test_set_expected_output)

    # else:
    #     print("Invalid training method")
    #     exit(1)

    # print(f"Error: {error}")
    # print(f"Predictions: {predictions}")
    # print(f"Expected output: {test_set_expected_output}")
    # # [0 0.3 0.5 1]
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    confusion_matrix = generate_confusion_matrix(
        test_set_expected_output, predictions, classes)

    print_metrics(confusion_matrix)
