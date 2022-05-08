import argparse
import numpy as np

from ..metrics.confusion_matrix import generate_confusion_matrix
from ..metrics import print_metrics, Metrics
from ..neural_network import NeuralNetwork
from ..metrics.holdout_validation import holdout_eval
from ..metrics.k_fold_cross_validation import k_fold_cross_validation_eval
from ..main import get_dataset, parse_config_file


def get_activation_function(activation_function):
    if activation_function == "non-linear":
        return "tanh"
    if activation_function == "linear":
        return "identity"


def parse_dataset(dataset_file_name):
    dataset = np.array([[]])

    with open(dataset_file_name, 'r') as dataset_file:
        for line in dataset_file:
            data = np.array([float(i) for i in line.split()])
            dataset = np.append(dataset, data).reshape(-1, len(data))

    return dataset


def get_epoch_metrics():
    return lambda scaled_predictions, expected_output: Metrics(0, 0, 0, 0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--function", type=str,
                        default="linear", help="Activation function to use with perceptron.\n Value may be 'linear' or 'non-linear'", dest="activation_function", required=False)
    parser.add_argument("--split_method", help="Method to split input data into train and test set.\n Use 'all' to use the entire dataset for training, 'holdout' for holdout validation, 'k-fold' for k-fold cross validation.",
                        dest='training_method', required=False, default='all')
    parser.add_argument("--k", help="Number of folds to use for k-fold cross validation.",
                        dest='k', required=False, default=10)
    parser.add_argument("--ratio", help="Ratio of training data to use for holdout validation.",
                        dest='ratio', required=False, default=0.9)

    args = parser.parse_args()

    input_dataset = parse_dataset('./TP3/datasets/ej2/input.txt')
    expected_output = parse_dataset('./TP3/datasets/ej2/output.txt')

    activation_function = get_activation_function(args.activation_function)
    training_method = args.training_method
    k = int(args.k)
    training_ratio = float(args.ratio)

    config_file_name = "./TP3/config.json"

    problem, dataset_path, output_dataset_path, metrics_output_path, prediction_output_path, network_parameters, training_parameters, other_parameters = parse_config_file(
        config_file_name)

    train_input_dataset, test_input_dataset, expected_output = get_dataset(
        problem, dataset_path, output_dataset_path, **other_parameters)

    def neural_network_supplier(): return NeuralNetwork(**network_parameters,
                                                        metrics_output_path=metrics_output_path, prediction_output_path=prediction_output_path)

    neural_network = neural_network_supplier()

    if training_method == 'all':
        test_set_input = input_dataset
        test_set_expected_output = expected_output
        neural_network.train(
            train_input_dataset, expected_output, get_epoch_metrics_fn=get_epoch_metrics(), **training_parameters, verbose=False)

        error, predictions = neural_network.test(
            test_set_input, test_set_expected_output)

    elif training_method == 'holdout':
        holdout_sets = holdout_eval(input_dataset, expected_output, neural_network,
                                           training_ratio=training_ratio, get_epoch_metrics_fn=get_epoch_metrics(), training_parameters=training_parameters, verbose=False)

        test_set_input = holdout_sets['test_set'][0]
        test_set_expected_output = holdout_sets['test_set'][1]

        error, predictions = neural_network.test(
            test_set_input, test_set_expected_output)

    elif training_method == 'k-fold':
        average_error, min_error_sets, neural_network = k_fold_cross_validation_eval(
            input_dataset, expected_output, neural_network_supplier, k=k, get_epoch_metrics_fn=get_epoch_metrics(), training_parameters=training_parameters, verbose=False)

        test_set_input = min_error_sets['test_set'][0]
        test_set_expected_output = min_error_sets['test_set'][1]

        error, predictions = neural_network.test(
            test_set_input, test_set_expected_output)

    else:
        print("Invalid training method")
        exit(1)

    print(f"Error: {error}")
    print(f"Predictions: {predictions}")
    print(f"Expected output: {test_set_expected_output}")
