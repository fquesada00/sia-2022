from io import FileIO
import random
import numpy as np

from typing import List

from .Layer import Layer
from .constants import BIAS_NEURON_ACTIVATION
from .ActivationFunction import ActivationFunction


class NeuralNetwork:
    def __init__(self, hidden_sizes: list, input_size: int, output_size: int, bias: float, activation_function_str: str, prediction_output_path: str = None, beta: float = 1):
        self.activation_function = activation_function_str
        self.bias = bias
        self.prediction_output_path = prediction_output_path
        activation_function = ActivationFunction(activation_function_str)
        # Calculate min and max values of the activation function of the output layer
        self.activation_min = activation_function.min()
        self.activation_max = activation_function.max()
        self.initialize_layers(
            input_size, output_size, hidden_sizes, beta)

    def initialize_layers(self, input_size: int, output_size: int, hidden_sizes: list, beta: float = 1):
        self.layers: List[Layer] = []
        self.layers.append(
            Layer(input_size, self.activation_function, beta))

        for layer_size in ([*hidden_sizes, output_size]):
            layer = Layer(layer_size, self.activation_function, beta)
            layer.connect_to(self.layers[-1], self.bias,)
            self.layers.append(layer)

    def propagate_forward(self, input_data: np.ndarray):

        lower_activation = input_data

        self.layers[0].activations = lower_activation

        for layer in self.layers[1:-1]:
            layer_activation = layer.propagate_forward(
                lower_activation)

            lower_activation = layer_activation

        return self.layers[-1].propagate_forward(
            lower_activation, True)

    def propagate_backward(self, output_delta: np.ndarray):
        # Start from the output layer
        upper_weights = self.layers[-1].weights
        upper_delta = output_delta

        # Set delta for output layer
        self.layers[-1].delta = upper_delta

        # Update weights for output layer
        self.layers[-1].update_accum_adjustment(
            self.layers[-2].activations, self.learning_rate)

        # Iterate over the hidden layers in reverse order
        for index in reversed(range(1, len(self.layers) - 1)):
            layer_delta = self.layers[index].propagate_backward(
                upper_weights, upper_delta)

            self.layers[index].update_accum_adjustment(
                self.layers[index - 1].activations, self.learning_rate)

            upper_delta = layer_delta
            upper_weights = self.layers[index].weights

    def update_weights(self,momentum):
        for layer in self.layers[1:]:
            layer.update_weights(momentum)

    def train(self, input_dataset: np.ndarray, expected_output: np.ndarray, learning_rate: float, batch_size: int, epochs: int = 1, tol: float = 1e-5, momentum: float = 0.9, verbose: bool = False):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        output_file = None
        if (self.prediction_output_path is not None):
            output_file = open(self.prediction_output_path, "w")
            self.save_output_weights_shape(output_file)

        dataset_length = len(input_dataset)
        batch_iteration = 0

        # Calculate min and max values of the expected output for normalization
        self.train_set_min = np.min(np.array(expected_output))
        self.train_set_max = np.max(np.array(expected_output))

        output_layer = self.layers[-1]

        # Iterate over the input data (the data set)
        current_epoch = 0
        output_delta = 1
        error = 1
        metrics_by_epoch = []

        while current_epoch < epochs and error > tol:

            current_epoch += 1
            random_indexes = random.sample(
                range(0, dataset_length), dataset_length)

            for index in random_indexes:
                output = self.predict(input_dataset[index])
                scaled_expected_output = self.scale(
                    expected_output[index], self.train_set_min, self.train_set_max, self.activation_min, self.activation_max)

                output_delta = output_layer.activation_function.derivative(
                    output_layer.excitations) * (scaled_expected_output - output)

                self.propagate_backward(output_delta)

                batch_iteration += 1

                if batch_iteration == self.batch_size:
                    # Print weights to file
                    if (self.prediction_output_path is not None):
                        self.save_output_weights(output_file)

                    # Calculate error scaling expected outputs to be in range of activation function image
                    scaled_expected_output = self.scale(
                        expected_output, self.train_set_min, self.train_set_max, self.activation_min, self.activation_max)
                    predictions = []

                    for input_data in input_dataset:
                        predictions.append(self.predict(input_data))

                    error = self.error(scaled_expected_output, predictions)

                    if error < tol:
                        break

                    self.update_weights(momentum)
                    # self.print_weights() 

                    batch_iteration = 0

                self.save_output_weights(output_file)

                print(f'error: {error}')

            # Counting epochs as full batches
            self.get_epoch_metrics(input_dataset, expected_output)
        print(f'error: {error}')

    def error(self, expected_output, predictions):

        accumulated_sum = 0

        for i in range(len(predictions)):
            expected = expected_output[i, :]
            error = expected - predictions[i]
            accumulated_sum += np.dot(error, error)

        accumulated_sum *= 0.5
        return accumulated_sum
        # return 0.5 * np.sum(np.square(expected_output.T - np.array([prediction for prediction in predictions])))

    def get_epoch_metrics(self, input_dataset: np.ndarray, expected_output: np.ndarray):
        # Calculate error scaling expected outputs to be in range of activation function image
        scaled_expected_output = self.scale(
            expected_output, self.train_set_min, self.train_set_max, self.activation_min, self.activation_max)
        predictions = []

        for input_data in input_dataset:
            predictions.append(self.predict(input_data))

        error = self.error(scaled_expected_output, predictions)

    def predict(self, input_data: np.ndarray):
        input_data_with_bias = np.insert(input_data, 0, BIAS_NEURON_ACTIVATION)
        return self.propagate_forward(input_data_with_bias)

    def test(self, input_dataset, expected_output):
        predictions = np.array([[]])

        for input_data in input_dataset:
            prediction = self.predict(input_data)
            predictions = np.append(
                predictions, prediction).reshape(-1, len(prediction))

        print(f"predictions without scaling: {predictions}")

        scaled_predictions = self.scale(
            predictions, self.activation_min, self.activation_max, self.train_set_min, self.train_set_max)

        print(f"scaled predictions: {scaled_predictions}")
        return self.error(expected_output, scaled_predictions), scaled_predictions

    def scale(self, x: np.ndarray, from_min: float, from_max: float, to_min: float, to_max: float):
        if from_min is None or from_max is None or to_min is None or to_max is None:
            return x
        return ((x - from_min) / (from_max - from_min)) * (to_max - to_min) + to_min

    def print_weights(self):
        for index, layer in enumerate(self.layers[1:]):
            print(f'Layer #{index + 1}: {layer.weights}')

    def save_output_weights(self, file: FileIO):
        if file is None:
            return

        newline = '\n'

        output_weights = self.layers[-1].weights

        file.write(
            f'{newline.join([f"{self.neuron_to_string(neuron)}" for neuron in self.layers[-1].weights])}{newline}')

    def save_output_weights_shape(self, file: FileIO):
        newline = '\n'
        tab = '\t'

        output_weights = self.layers[-1].weights

        file.write(
            f'{tab.join([str(dim) for dim in output_weights.shape])}{newline}')

    def neuron_to_string(self, neuron: np.ndarray):
        return '\t'.join([str(weight) for weight in neuron])
