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


def parse_number_input_dataset(dataset_file_name, add_noise=False, noise_level=0.1):
    dataset = np.zeros((10, 5 * 7), dtype=int)
    with open(dataset_file_name, 'r') as dataset_file:
        dataset_file.seek(0)
        data_index = 0
        data = []
        for i, line in enumerate(dataset_file):
            line = line.split()

            if add_noise:
                # Add noise to line
                for j, element in enumerate(line):
                    if np.random.rand() > 1 - noise_level:
                        line[j] = '1' if element == '0' else '0'

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

    # input_dataset, expected_output, noise_input = get_dataset(
    #     args.dataset, "TP3/ej3/datasets/input.txt", "TP3/ej3/datasets/parity_expected_output.txt" if args.dataset == "parity" else "TP3/ej3/datasets/numbers_expected_output.txt")

    neural_network = NeuralNetwork(hidden_sizes=[30, 20], input_size=35, output_size=10, learning_rate=1,
                                   bias=0.5, activation_function_str=activation_function, batch_size=1, beta=1)
    # test_set_input = input_dataset if args.dataset == "xor" else noise_input
    # test_set_expected_output = expected_output
    # neural_network.train(input_dataset, expected_output, epochs=50)

    # error, predictions = neural_network.test(
    #     test_set_input, test_set_expected_output)
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

    confusion_matrix = generate_confusion_matrix(
        test_set_expected_output, predictions, args.dataset)

    print_metrics(confusion_matrix)
