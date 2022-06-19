import time
import numpy as np
from models.Autoencoder import Autoencoder, create_autoencoder
from models.Model import Model
from utils import to_bin_array, to_raw_dataset, add_noise, plot_denoiser
from utils.dataset import font_1, font_2, font_3


def denoising_autoencoder(dataset: list[list[int]], encoder_layers: list[int], decoder_layers: list[int],
                          encoder_activations: list[str], decoder_activations: list[str], optimizer: str,
                          epochs: int, noise_type: str, noise_amount: float, noise_samples: int = 1) -> Model:
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
    input_size = len(dataset[0])
    encoder, decoder = create_autoencoder(
        input_size, encoder_layers, decoder_layers, encoder_activations, decoder_activations)
    autoencoder = Autoencoder(encoder, decoder, optimizer)

    # Create noisy dataset
    noisy_dataset = [add_noise(np.copy(bitmap), noise_type, noise_amount, seed=i)
                     for i, bitmap in enumerate(np.repeat(dataset, noise_samples, axis=0))]

    # Train autoencoder
    autoencoder.fit(noisy_dataset, dataset, epochs)

    return autoencoder


if __name__ == "__main__":
    labelled_dataset = font_2

    # Set parameters
    raw_dataset = np.array([data.flatten() for data in map(
        to_bin_array, to_raw_dataset(labelled_dataset))])

    input_size = len(raw_dataset[0])
    latent_space_size = 45
    encoder_layers = [40, latent_space_size]
    decoder_layers = [40, input_size]
    encoder_activations = ["relu", "relu", "logistic"]
    decoder_activations = ["relu", "relu", "logistic"]

    # Create and train autoencoder
    start = time.time()
    dae = denoising_autoencoder(raw_dataset[:5], encoder_layers, decoder_layers, encoder_activations,
                                decoder_activations, optimizer='powell', epochs=10, noise_type="gauss", noise_amount=0.5, noise_samples=5)
    end = time.time()
    print("Training time: ", end - start)

    # Test autoencoder
    test_set = [add_noise(char.copy(), "gauss", 0.5)
                for char in raw_dataset[:5]]

    predictions = [dae(noisy_char)
                   for noisy_char in test_set[:5]]

    plot_denoiser(test_set, predictions, raw_dataset[:5], labelled_dataset)
