
import time
from models.Autoencoder import Autoencoder, create_autoencoder
from models.Model import Model
from models.Dense import Dense
from models.Input import Input
from utils import plot, to_bin_array, to_raw_dataset, add_noise, plot_5n_letters, plot_denoiser, plot_latent_space
from utils.dataset import font_1, font_2, font_3
import numpy as np


if __name__ == "__main__":
    font = font_2
    labelled_dataset = font

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
    autoencoder.fit(raw_dataset, raw_dataset, epochs=2)
    end = time.time()

    print("Training time: ", end - start)

    # Test autoencoder
    predictions = [autoencoder(char).reshape((7, 5))
                   for char in raw_dataset[:5]]

    # Plot 5 predictions
    plot_5n_letters(predictions, labelled_dataset, n=1)

    # plot(prediction.reshape((7, 5)))

    # Plot latent space
    plot_latent_space(encoder, labelled_dataset)
