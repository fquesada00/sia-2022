
import time
from models.Autoencoder import Autoencoder, create_autoencoder
from models.Model import Model
from models.Dense import Dense
from models.Input import Input
from utils import to_bin_array, to_raw_dataset, plot_5n_letters, plot_latent_space
from utils.dataset import font_1, font_2, font_3
import numpy as np
import argparse


if __name__ == "__main__":
    np.random.seed(4)
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent", type=int, default=2,
                        help="Latent space dimension")
    parser.add_argument("--architecture", type=str,
                        default="[]", help="Architecture")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")

    parser.add_argument("--read_weights", action="store_true",
                        help="Read weights from file")

    args = parser.parse_args()
    latent_space = args.latent
    architecture = eval(args.architecture)
    epochs = args.epochs
    read_weights = args.read_weights

    font = font_2
    labelled_dataset = font

    raw_dataset = np.array([data.flatten() for data in map(
        to_bin_array, to_raw_dataset(labelled_dataset))])

    raw_dataset = raw_dataset
    reversed_architecture = architecture.copy()
    reversed_architecture.reverse()
    # Create autoencoder
    input_size = len(raw_dataset[0])
    latent_space_size = latent_space
    encoder_layers = architecture + [latent_space_size]
    decoder_layers = reversed_architecture + [input_size]
    encoder_activations = ["relu"] * len(architecture) + ["identity"]
    decoder_activations = ["relu"] * len(reversed_architecture) + ["logistic"]
    print(f"Encoder layers: {encoder_layers}")
    print(f"Decoder layers: {decoder_layers}")
    encoder, decoder = create_autoencoder(
        input_size, encoder_layers, decoder_layers, encoder_activations, decoder_activations,)

    autoencoder = Autoencoder(encoder, decoder, optimizer='powell')

    start = time.time()

    weights_filename = f"data/weights_{'-'.join(map(str, architecture))}_{latent_space}_{epochs}.txt"

    error_filename = f"data/error_{'-'.join(map(str, architecture))}_{latent_space}_{epochs}.txt"

    if read_weights:
        weights = np.loadtxt(weights_filename)
        autoencoder.load_weights(weights)

    autoencoder.fit(raw_dataset, raw_dataset, epochs=epochs,
                    weights_filename=weights_filename,
                    error_filename=error_filename)
    end = time.time()

    print("Training time: ", end - start)

    # Test autoencoder
    predictions = [autoencoder(char).reshape((7, 5))
                   for char in raw_dataset]

    # Plot 5 predictions
    plot_5n_letters(predictions, labelled_dataset, n=6)

    # plot(prediction.reshape((7, 5)))

    # Plot latent space
    plot_latent_space(encoder, labelled_dataset)
