import json
from matplotlib import pyplot as plt

import numpy as np

from .metrics.confusion_matrix import generate_confusion_matrix
from .metrics import generate_metrics
from .ej3.main import parse_number_input_dataset
from .neural_network import NeuralNetwork


def parse_number_array_file(path, array_size, dtype=float):
    dataset = np.zeros((1, array_size), dtype=dtype)

    with open(path) as f:
        line_count = 0

        for i, line in enumerate(f):
            line_count += 1
            dataset.resize(line_count, array_size)
            dataset[i] = np.array(line.split())

    return dataset


def get_default_dataset_paths(problem):
    if problem == "xor_simple":
        input_dataset_path = "./TP3/datasets/ej1/logic_input.txt"
        output_dataset_path = "./TP3/datasets/ej1/xor_output.txt"
    elif problem == "and":
        input_dataset_path = "./TP3/datasets/ej1/logic_input.txt"
        output_dataset_path = "./TP3/datasets/ej1/and_output.txt"
    elif problem == "parity" or problem == "parity_noise":
        input_dataset_path = "./TP3/datasets/ej3/number_input.txt"
        output_dataset_path = "./TP3/datasets/ej3/parity_output.txt"
    elif problem == "number" or problem == "number_noise":
        input_dataset_path = "./TP3/datasets/ej3/number_input.txt"
        output_dataset_path = "./TP3/datasets/ej3/number_output.txt"
    elif problem == "xor_multilayer":
        input_dataset_path = "./TP3/datasets/ej3/logic_input.txt"
        output_dataset_path = "./TP3/datasets/ej3/xor_output.txt"
    elif problem == "ej_2_linear" or problem == "ej_2_non_linear":
        input_dataset_path = "./TP3/datasets/ej2/input.txt"
        output_dataset_path = "./TP3/datasets/ej2/output.txt"

    return input_dataset_path, output_dataset_path

def check_valid_test_size(test_set: np.ndarray, test_size: int):
    print("Test size:", test_size)
    print("Test set size:", test_set.shape[0])
    if test_size < 0:
        raise ValueError("Test size must be greater than 0")
    if test_size > test_set.shape[0]:
        raise ValueError("Test size must be smaller than the test set size")


def get_dataset(problem: str, input_dataset_path: str, output_dataset_path: str, add_noise_to_number_dataset: bool = False, noise_level: float = 0.1, test_set_size: int = 0):
    if input_dataset_path == "":
        input_dataset_path = get_default_dataset_paths(problem)[0]

    if output_dataset_path == "":
        output_dataset_path = get_default_dataset_paths(problem)[1]

    if problem == "parity" or problem == "parity_noise":
        train_input_dataset = parse_number_input_dataset(
            input_dataset_path, add_noise=False)
        test_input_dataset = parse_number_input_dataset(
            input_dataset_path, add_noise=add_noise_to_number_dataset, noise_level=noise_level)
        expected_output = parse_number_array_file(
            output_dataset_path, 1, dtype=int)

    elif problem == "number" or problem == "number_noise":
        train_input_dataset = parse_number_input_dataset(
            input_dataset_path, add_noise=False)
        test_input_dataset = parse_number_input_dataset(
            input_dataset_path, add_noise=add_noise_to_number_dataset, noise_level=noise_level)
        expected_output = parse_number_array_file(
            output_dataset_path, 10, dtype=int)

    elif problem == "xor_simple" or problem == "xor_multilayer" or problem == "and":
        train_input_dataset = parse_number_array_file(
            input_dataset_path, 2, dtype=int)
        test_input_dataset = train_input_dataset
        expected_output = parse_number_array_file(
            output_dataset_path, 1, dtype=int)

    elif problem == "ej_2_linear" or problem == "ej_2_non_linear":
        train_input_dataset = parse_number_array_file(
            input_dataset_path, 3, dtype=float)
        test_input_dataset = train_input_dataset
        expected_output = parse_number_array_file(
            output_dataset_path, 1, dtype=float)

    check_valid_test_size(test_input_dataset, test_set_size)

    test_input_dataset = test_input_dataset[:test_set_size]
    test_expected_output_dataset = expected_output[:test_set_size]

    print("Train input dataset:")
    print(train_input_dataset)

    print("Test input dataset:")
    print(test_input_dataset)

    print("Output dataset:")
    print(expected_output)

    print("Test expected output dataset:")
    print(test_expected_output_dataset)

    return train_input_dataset, test_input_dataset, expected_output, test_expected_output_dataset


def get_epoch_metrics(predictions: np.ndarray, expected_output: np.ndarray, problem: str):
    epoch_confusion_matrix = generate_confusion_matrix(
        expected_output, predictions, problem)
    epoch_metrics = generate_metrics(epoch_confusion_matrix)

    return epoch_metrics


def get_epoch_metrics_by_problem(problem: str):
    if problem.startswith("ej_2"):
        return None
    return lambda predictions, scaled_expected_output: get_epoch_metrics(predictions, scaled_expected_output, problem)


def parse_config_file(config_file_name):
    with open(config_file_name) as config_file:
        config = json.load(config_file)

        problem = config['problem']
        input_dataset_path = config['input_dataset_path']
        output_dataset_path = config['output_dataset_path']
        metrics_output_path = config['metrics_output_path']
        prediction_output_path = config['prediction_output_path']
        use_optimal_parameters = config['use_optimal_parameters']

        if problem not in ["xor_simple", "and", "parity", "number", "xor_multilayer", "ej_2_linear", "ej_2_non_linear", "number_noise", "parity_noise"]:
            raise ValueError(
                "Invalid problem name used in config file. Valid options are: xor_simple, xor_multilayer, ej_2, parity, number, number_noise, parity_noise")

        if use_optimal_parameters:
            parameters = config["optimal_parameters"][problem]
        else:
            parameters = config["custom_parameters"]

        # rename activation_function key to activation_function_str for compatibility with neural_network.py
        if "activation_function" in parameters["network"]:
            parameters["network"]["activation_function_str"] = parameters["network"]["activation_function"]
            del parameters["network"]["activation_function"]

        return problem, input_dataset_path, output_dataset_path, metrics_output_path, prediction_output_path, parameters["network"], parameters["training"], parameters["other"]

def graph_number_with_parity_prediction(number: np.ndarray, prediction: float, expected_output: float):
    for i in range(len(number)):
        if number[i] == 0:
            number[i] = 1
        else:
            number[i] = 0

    # round to 5 decimals
    prediction = np.around(prediction, decimals=5)
    expected_output = np.around(expected_output, decimals=5)

    plt.figure(figsize=(5, 5))
    plt.imshow(number.reshape(7, 5), cmap='gray')
    plt.text(0, -1, f"Expected: {expected_output} - Predicted: {prediction}", bbox=dict(fill=False, edgecolor='red', linewidth=2))    


def graph_number_with_number_prediction(number: np.ndarray, prediction: np.ndarray, expected_output: np.ndarray):
    for i in range(len(number)):
        if number[i] == 0:
            number[i] = 1
        else:
            number[i] = 0

    # change prediction and expected output to a number
    prediction = np.argmax(prediction)
    expected_output = np.argmax(expected_output)

    plt.figure(figsize=(5, 5))
    plt.imshow(number.reshape(7, 5), cmap='gray')
    plt.text(0, -1, f"Expected: {expected_output} - Predicted: {prediction}", bbox=dict(fill=False, edgecolor='red', linewidth=2))    

    

if __name__ == '__main__':
    config_file_name = "./TP3/config.json"

    problem, input_dataset_path, output_dataset_path, metrics_output_path, prediction_output_path, network_parameters, training_parameters, other_parameters = parse_config_file(
        config_file_name)

    neural_network = NeuralNetwork(
        **network_parameters, metrics_output_path=metrics_output_path, prediction_output_path=prediction_output_path)

    train_input_dataset, test_input_dataset, expected_output, test_expected_output_dataset = get_dataset(
        problem, input_dataset_path, output_dataset_path, **other_parameters)

    neural_network.train(
        train_input_dataset, expected_output, get_epoch_metrics_fn=get_epoch_metrics_by_problem(problem), **training_parameters)

    error, predictions = neural_network.test(
        test_input_dataset, test_expected_output_dataset)

    print("Test input dataset: ", test_input_dataset)
    print("Error: ", error)
    print("Predictions: ", predictions)
    print("Test expected output: ", test_expected_output_dataset)

    if problem.startswith("parity"):
        for i in range(len(test_input_dataset)):
            # plt.figure(i)
            graph_number_with_parity_prediction(test_input_dataset[i], predictions[i][0], test_expected_output_dataset[i][0])
        
        plt.show()
    elif problem.startswith("number"):
        for i in range(len(test_input_dataset)):
            graph_number_with_number_prediction(test_input_dataset[i], predictions[i], test_expected_output_dataset[i])

        plt.show()
        
    
