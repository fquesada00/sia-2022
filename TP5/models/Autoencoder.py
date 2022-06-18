from pprint import pprint
import numpy as np
import numdifftools as nd

from models.OptimizerFunction import OptimizerFunction
from .Model import Model


class Autoencoder():

    def __init__(self, encoder: Model, decoder: Model, optimizer='adam'):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = OptimizerFunction(optimizer)

    def __call__(self, input_data):
        return self.decoder(self.encoder(input_data))

    def feed_forward(self, input_data: np.ndarray, encoder_weights: np.ndarray, decoder_weights: np.ndarray):
        previous_layer_activation = input_data
        layer_weight_index = 0

        for encoder_layer in self.encoder.layers[1:]:
            layer_weights_size = encoder_layer.weights.size
            previous_layer_activation = encoder_layer.forward(
                previous_layer_activation, encoder_weights[layer_weight_index:layer_weight_index+layer_weights_size])

            layer_weight_index += layer_weights_size

        # decoder forward pass
        layer_weight_index = 0

        for decoder_layer in self.decoder.layers[1:]:
            layer_weights_size = decoder_layer.weights.size

            previous_layer_activation = decoder_layer.forward(
                previous_layer_activation, decoder_weights[layer_weight_index:layer_weight_index+layer_weights_size])

            layer_weight_index += layer_weights_size

        return previous_layer_activation

    # Builds loss function from neural network using weights as parameters
    def build_loss_function(self, input_dataset: np.ndarray, target_dataset: np.ndarray):

        # weights are flattened
        def loss_function(weights, step=None):

            # convert flatten weights to layers
            total_encoder_weights = self.encoder.total_weights()
            encoder_weights = weights[:total_encoder_weights]
            decoder_weights = weights[total_encoder_weights:]
            predictions = np.array([self.feed_forward(
                input_data, encoder_weights, decoder_weights) for input_data in input_dataset])
            distances = predictions - target_dataset

            error = 0.5 * np.sum(distances ** 2)
            reg_term = 0.000005 * np.sum(abs(weights))
            loss = error + reg_term
            # print(f'Step {step}: {loss}')

            return loss

        return loss_function

    def update_weights(self, updated_weights: np.ndarray):
        # update encoder weights
        layer_weight_offset = 0
        for layer in [*self.encoder.layers[1:], *self.decoder.layers[1:]]:
            layer_weights_size = layer.weights.size

            layer.update_weights(
                updated_weights[layer_weight_offset:layer_weight_offset+layer_weights_size])

            layer_weight_offset += layer_weights_size

    def fit(self, input_dataset: np.ndarray, target_dataset: np.ndarray, epochs: int = 10, learning_rate: float = 0.01):
        # flatten weights
        flattened_weights = []

        for layer in [*self.encoder.layers[1:], *self.decoder.layers[1:]]:
            flattened_weights = np.append(
                flattened_weights, layer.weights.flatten())

        # build loss function
        loss_function = self.build_loss_function(
            input_dataset, target_dataset)

        # build optimizer function
        optimal_weights = self.optimizer(
            loss_function, flattened_weights, step_size=learning_rate, num_iters=epochs)

        print(f"loss ===> {loss_function(optimal_weights)}")

        self.update_weights(optimal_weights)
