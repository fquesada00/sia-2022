from io import FileIO
import random
import numpy as np

from typing import List

from .Layer import Layer
from .constants import BIAS_NEURON_ACTIVATION


class NeuralNetwork:
    def __init__(self, hidden_sizes: list, input_size: int, output_size: int, learning_rate: float, bias: float, activation_function: str, batch_size=1, output_file_name: str = None):
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.bias = bias
        self.output_file_name = output_file_name
        self.initialize_layers(input_size, output_size, hidden_sizes)

    def initialize_layers(self, input_size: int, output_size: int, hidden_sizes: list):
        self.layers: List[Layer] = []
        self.layers.append(
            Layer(input_size, self.activation_function))

        for layer_size in ([*hidden_sizes, output_size]):
            layer = Layer(layer_size, self.activation_function)
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

    def update_weights(self):
        for layer in self.layers[1:]:
            layer.update_weights()

    def train(self, input_dataset: np.ndarray, expected_output: np.ndarray, epochs: int = 1, tol: float = 1e-5):
        output_file = None
        if (self.output_file_name is not None):
            output_file = open(self.output_file_name, "w")
            self.save_output_weights_shape(output_file)

        dataset_length = len(input_dataset)
        batch_iteration = 0

        # Calculate min and max values of the expected output for normalization
        from_min = np.min(np.array(expected_output))
        from_max = np.max(np.array(expected_output))

        # Calculate min and max values of the activation function of the output layer
        output_layer = self.layers[-1]
        to_min = output_layer.activation_function.min()
        to_max = output_layer.activation_function.max()

        # Iterate over the input data (the data set)
        current_epoch = 0
        output_delta = 1
        error = self.error(input_dataset, expected_output)
        
        while current_epoch < epochs and error > tol:

            current_epoch += 1
            random_indexes = random.sample(range(0, dataset_length), dataset_length)
            for index in random_indexes:
                output = self.predict(input_dataset[index])

                # Normalize the output
                normalized_output = self.normalize(
                    output, from_min, from_max, to_min, to_max)

                output_delta = output_layer.activation_function.derivative(
                    output_layer.excitations) * (expected_output[index] - normalized_output)

                self.propagate_backward(output_delta)

                
                batch_iteration += 1
                    
                error = self.error(input_dataset, expected_output)

                if error < tol:
                    break

                if batch_iteration == self.batch_size:
                    # Print weights to file
                    if (self.output_file_name is not None):
                        self.save_output_weights(output_file)

                    self.update_weights()
                    
                    batch_iteration = 0

                self.save_output_weights(output_file)

                
                print(f'error: {error}')
        
        print(f'error: {error}')

                    
    def error(self, input_dataset, expected_output, predictions=None):
        if predictions is None:
            predictions = []
        for input_data in input_dataset:
            predictions.append(self.predict(input_data))
        accumulated_sum = 0
        print(predictions)
        for i in range(len(predictions)):
            accumulated_sum += (expected_output.T[0][i] - predictions[i]) ** 2
        return accumulated_sum / 2
        return 0.5 * np.sum(np.square(expected_output.T - np.array([prediction for prediction in predictions])))

    def predict(self, input_data: np.ndarray):
        input_data_with_bias = np.insert(input_data, 0, BIAS_NEURON_ACTIVATION)
        return self.propagate_forward(input_data_with_bias)

    def test(self, input_dataset, expected_output):
        predictions = []
        return self.error(input_dataset,expected_output, predictions), predictions
        

    def normalize(self, x: np.ndarray, from_min: float, from_max: float, to_min: float, to_max: float):
        if to_min is None or to_max is None:
            return x
        return (x - from_min) / (from_max - from_min) * (to_max - to_min) + to_min

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
