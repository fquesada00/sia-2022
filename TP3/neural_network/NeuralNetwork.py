from io import FileIO
import numpy as np

from typing import List

from ..metrics.Metrics import Metrics
from .Layer import Layer
from .constants import BIAS_NEURON_ACTIVATION
from .ActivationFunction import ActivationFunction


class NeuralNetwork:
    def __init__(self, hidden_sizes: list, input_size: int, output_size: int, bias: float, activation_function_str: str, prediction_output_path: str = None, metrics_output_path: str = None, beta: float = 1):
        self.activation_function = activation_function_str
        self.bias = bias

        if prediction_output_path == '':
            self.prediction_output_path = None
        else:
            self.prediction_output_path = prediction_output_path

        if metrics_output_path == '':
            self.metrics_output_path = None
        else:
            self.metrics_output_path = metrics_output_path

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

    def update_weights(self, momentum):
        for layer in self.layers[1:]:
            layer.update_weights(momentum)

    def update_learning_rate(self, delta_error: float, alpha, beta):
        print('UPDATE ETA')
        if(delta_error < 0):
            self.learning_rate = self.learning_rate + alpha
        elif delta_error > 0:
            self.learning_rate = self.learning_rate - beta*self.learning_rate

    def train(self, input_dataset: np.ndarray, expected_output: np.ndarray, learning_rate: float = 1,
              batch_size: int = 1, epochs: int = 1, tol: float = 1e-5, momentum: float = 0.9, verbose: bool = False, alpha: float = 0.05, beta: float = 0.05, k: int = 0, get_epoch_metrics_fn=None, use_adaptive_learning_rate: bool = False,
              test_input_dataset: np.ndarray = None, test_expected_output: np.ndarray = None, test_metrics_output_path: str = None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        output_file = None

        if (self.prediction_output_path is not None):
            output_file = open(self.prediction_output_path, "a")
            self.save_output_weights_shape(output_file)

        if (self.metrics_output_path is not None):
            metrics_file = open(self.metrics_output_path, "a")

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
        consistent_error_variation = 0

        while current_epoch < epochs and error > tol:
            print("Epoch: ", current_epoch)
            current_epoch += 1
            random_indexes = np.array(range(0, dataset_length))
            np.random.shuffle(random_indexes)

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
                    if self.prediction_output_path is not None:
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

                    if use_adaptive_learning_rate:
                        predictions = []

                        for input_data in input_dataset:
                            predictions.append(self.predict(input_data))

                        new_error = self.error(
                            scaled_expected_output, predictions)

                        delta_error = new_error - error
                        delta_error_sign = np.sign(delta_error)

                        if (consistent_error_variation > 0 and delta_error_sign < 0) or consistent_error_variation < 0 and delta_error_sign > 0:
                            consistent_error_variation = 0
                        else:
                            consistent_error_variation += delta_error_sign

                        if abs(consistent_error_variation) == k:
                            self.update_learning_rate(
                                delta_error_sign, alpha, beta)

                            consistent_error_variation = 0

                    # self.print_weights()

                    batch_iteration = 0

                self.save_output_weights(output_file)

                if verbose:
                    print(f'error: {error}')

            # save last epoch metrics
            if(self.metrics_output_path is not None and get_epoch_metrics_fn is not None):
                print("Saving metrics...")
                predictions = np.empty((0, output_layer.size))

                for input_data in input_dataset:
                    predictions = np.append(
                        predictions, self.predict(input_data)).reshape(-1, output_layer.size)

                scaled_predictions = self.scale(
                    predictions, self.activation_min, self.activation_max, self.train_set_min, self.train_set_max)

                # Using confusion matrix to calculate metrics
                epoch_metrics = get_epoch_metrics_fn(
                    scaled_predictions, expected_output)

                scaled_predictions_epoch_error = self.error(
                    expected_output, scaled_predictions)

                # Calculate error scaling expected outputs to be in range of activation function image
                scaled_expected_output = self.scale(
                    expected_output, self.train_set_min, self.train_set_max, self.activation_min, self.activation_max)

                scaled_expected_output_epoch_error = self.error(
                    scaled_expected_output, predictions)

                self.save_epoch_metrics(
                    metrics_file, epoch_metrics, scaled_predictions_epoch_error, scaled_expected_output_epoch_error, current_epoch)


            if test_input_dataset is not None and test_expected_output is not None and test_metrics_output_path is not None:
                print("Saving test metrics...")
                self.test(test_input_dataset, test_expected_output, test_metrics_output_path, get_epoch_metrics_fn, epoch_number=current_epoch)

        if (self.prediction_output_path is not None):
            output_file.close()

        if (self.metrics_output_path is not None):
            metrics_file.close()

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

    def save_epoch_metrics(self, metrics_file, epoch_metrics: Metrics, scaled_predictions_epoch_error: float, scaled_expected_output_epoch_error: float, current_epoch: int):
        metrics_file.write(f'{current_epoch}\n')

        for metric in epoch_metrics:
            if type(metric) == np.ndarray:
                for value in metric:
                    metrics_file.write(f'{value:.5f} ')

                metrics_file.write('\n')
            else:
                metrics_file.write(f'{metric:.5f}\n')

        metrics_file.write(
            f'{scaled_predictions_epoch_error:.5f}\n{scaled_expected_output_epoch_error:.5f}\n')

    def predict(self, input_data: np.ndarray):
        input_data_with_bias = np.insert(input_data, 0, BIAS_NEURON_ACTIVATION)
        return self.propagate_forward(input_data_with_bias)

    def test(self, input_dataset, expected_output, metrics_output_filename=None, get_epoch_metrics_fn=None, epoch_number=-1):
        predictions = np.array([[]])
        for input_data in input_dataset:
            prediction = self.predict(input_data)
            predictions = np.append(
                predictions, prediction).reshape(-1, len(prediction) if type(prediction) == np.ndarray else 1)

        scaled_predictions = self.scale(
            predictions, self.activation_min, self.activation_max, self.train_set_min, self.train_set_max)

        if metrics_output_filename is not None and get_epoch_metrics_fn is not None:
            # save last epoch metrics
            metrics_file = open(metrics_output_filename, "a")
            print("Saving metrics...")
            # Using confusion matrix to calculate metrics
            epoch_metrics = get_epoch_metrics_fn(
                scaled_predictions, expected_output)

            scaled_predictions_epoch_error = self.error(
                expected_output, scaled_predictions)

            # Calculate error scaling expected outputs to be in range of activation function image
            scaled_expected_output = self.scale(
                expected_output, self.train_set_min, self.train_set_max, self.activation_min, self.activation_max)

            scaled_expected_output_epoch_error = self.error(
                scaled_expected_output, predictions)

            self.save_epoch_metrics(
                metrics_file, epoch_metrics, scaled_predictions_epoch_error, scaled_expected_output_epoch_error, self.epochs if epoch_number == -1 else epoch_number)

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
