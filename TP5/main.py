
import time
from utils import plot_latent_space
from models.Autoencoder import Autoencoder
from models.Model import Model
from models.Dense import Dense
from models.Input import Input
from utils import plot, to_bin_array, to_raw_dataset, add_noise
from utils.dataset import font_1, font_2, font_3
import numpy as np


def create_autoencoder(input_size: int, encoder_layers: list[int], decoder_layers: list[int],
                       encoder_activations: list[str], decoder_activations: list[str]) -> tuple[Model, Model]:
    # Create encoder
    input_layer = Input(shape=(input_size,), name="Encoder Input")
    prev_layer = input_layer

    for i, hidden_size in enumerate(encoder_layers[:-1]):
        hidden_layer = Dense(
            hidden_size, activation=encoder_activations[i], name=f"Encoder Hidden {i}")(prev_layer)
        prev_layer = hidden_layer

    encoder_output = Dense(
        encoder_layers[-1], activation=encoder_activations[-1], name="Encoder Output")(prev_layer)

    encoder = Model(input_layer=input_layer,
                    output_layer=encoder_output, name="Encoder")

    # Create decoder
    input_layer = Input(shape=(encoder_layers[-1],), name="Decoder Input")
    prev_layer = input_layer

    for i, hidden_size in enumerate(decoder_layers[:-1]):
        hidden_layer = Dense(
            hidden_size, activation=decoder_activations[i], name=f"Decoder Hidden {i}")(prev_layer)
        prev_layer = hidden_layer

    decoder_output = Dense(
        decoder_layers[-1], activation=decoder_activations[-1], name="Decoder Output")(prev_layer)

    decoder = Model(input_layer=input_layer,
                    output_layer=decoder_output, name="Decoder")

    return encoder, decoder


def denoising_autoencoder(dataset: list[list[int]], encoder_layers: list[int], decoder_layers: list[int],
                          encoder_activations: list[str], decoder_activations: list[str], optimizer: str,
                          epochs: int, noise_type: str, noise_amount: float) -> Model:
    """
    Create and train a denoising autoencoder.

    Parameters
    ----------
    dataset : list[list[int]]
        Dataset to train the autoencoder on.
    encoder_layers : list[int]
        List of hidden layers sizes for the encoder.
    decoder_layers : list[int]
        List of hidden layers sizes for the decoder.
    encoder_activations : list[str]
        List of activations for the encoder.
    decoder_activations : list[str]
        List of activations for the decoder.
    optimizer : str
        Optimizer to use for training.
    epochs : int
        Number of epochs to train the model for.
    noise_type : str
        Type of noise to add to the dataset.
    noise_amount : float
        Amount of noise to add to the dataset.
    """

    # Create autoencoder
    input_size = len(dataset[0][0])
    encoder, decoder = create_autoencoder(
        input_size, encoder_layers, decoder_layers, encoder_activations, decoder_activations)
    autoencoder = Autoencoder(encoder, decoder, optimizer)

    # Create noisy dataset
    noisy_dataset = [add_noise(np.copy(bitmap), noise_type, noise_amount)
                     for bitmap in dataset]

    # Train autoencoder
    autoencoder.fit(noisy_dataset, dataset, epochs)

    return autoencoder


if __name__ == "__main__":
    font = font_2
    labelled_dataset = font[:5]
    raw_dataset = np.array([data.flatten() for data in map(
        to_bin_array, to_raw_dataset(labelled_dataset))])

    # Create autoencoder
    input_size = len(raw_dataset[0])
    latent_space_size = 2
    encoder_layers = [20, 10, latent_space_size]
    decoder_layers = [10, 20, input_size]
    encoder_activations = ["relu", "relu", "logistic"]
    decoder_activations = ["relu", "relu", "logistic"]

    encoder, decoder = create_autoencoder(
        input_size, encoder_layers, decoder_layers, encoder_activations, decoder_activations)

    autoencoder = Autoencoder(encoder, decoder, optimizer='powell')

    start = time.time()
    autoencoder.fit(raw_dataset, raw_dataset, epochs=10)
    end = time.time()

    print("Training time: ", end - start)
    # Test autoencoder
    prediction = autoencoder(raw_dataset[3])

    plot(prediction.reshape((7, 5)))

    # Plot latent space
    plot_latent_space(encoder, labelled_dataset)
