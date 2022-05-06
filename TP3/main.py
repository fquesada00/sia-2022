import json

import numpy as np
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


def get_dataset(problem: str, input_dataset_path: str, output_dataset_path: str, add_noise_to_number_dataset: bool = False, noise_level: float = 0.1):
    if input_dataset_path == "":
        input_dataset_path = get_default_dataset_paths(problem)[0]

    if output_dataset_path == "":
        output_dataset_path = get_default_dataset_paths(problem)[1]

    if problem == "parity" or problem == "parity_noise":
        input_dataset = parse_number_input_dataset(
            input_dataset_path, add_noise=add_noise_to_number_dataset, noise_level=noise_level)
        expected_output = parse_number_array_file(
            output_dataset_path, 1, dtype=int)

    elif problem == "number" or problem == "number_noise":
        input_dataset = parse_number_input_dataset(
            input_dataset_path, add_noise=add_noise_to_number_dataset, noise_level=noise_level)
        expected_output = parse_number_array_file(
            output_dataset_path, 10, dtype=int)

    elif problem == "xor_simple" or problem == "xor_multilayer" or problem == "and":
        input_dataset = parse_number_array_file(
            input_dataset_path, 2, dtype=int)
        expected_output = parse_number_array_file(
            output_dataset_path, 1, dtype=int)

    elif problem == "ej_2_linear" or problem == "ej_2_non_linear":
        input_dataset = parse_number_array_file(
            input_dataset_path, 3, dtype=float)
        expected_output = parse_number_array_file(
            output_dataset_path, 1, dtype=float)

    print("Input dataset:")
    print(input_dataset)

    print("Output dataset:")
    print(expected_output)

    return input_dataset, expected_output


def parse_config_file(config_file_name):
    with open(config_file_name) as config_file:
        config = json.load(config_file)

        problem = config['problem']
        input_dataset_path = config['input_dataset_path']
        output_dataset_path = config['output_dataset_path']
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

        return problem, input_dataset_path, output_dataset_path, prediction_output_path, parameters["network"], parameters["training"], parameters["other"]


if __name__ == '__main__':
    config_file_name = "./TP3/config.json"

    problem, input_dataset_path, output_dataset_path, prediction_dataset_path, network_parameters, training_parameters, other_parameters = parse_config_file(
        config_file_name)

    neural_network = NeuralNetwork(**network_parameters)

    input_dataset, expected_output = get_dataset(
        problem, input_dataset_path, output_dataset_path, **other_parameters)

    neural_network.train(
        input_dataset, expected_output, **training_parameters)

    error, predictions = neural_network.test(input_dataset, expected_output)

    print("Error: ", error)
    print("Predictions: ", predictions)
