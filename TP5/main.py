
import time
from utils import plot_latent_space
from models.Autoencoder import Autoencoder
from models.Model import Model
from models.Dense import Dense
from models.Input import Input
from utils import plot, to_bin_array, to_raw_dataset
from utils.dataset import font_1, font_2, font_3
import numpy as np

if __name__ == "__main__":
    font = font_2
    labelled_dataset = font[:5]
    raw_dataset = [data.flatten() for data in map(
        to_bin_array, to_raw_dataset(labelled_dataset))]
    # input_shape = dataset[0].
    # Create encoder and decoder
    x = Input(shape=(35,), name="Encoder Input")
    hidden_encoder_layer_1 = Dense(
        20, activation="relu", name="Hidden layer")(x)
    hidden_encoder_layer_2 = Dense(
        10, activation="relu", name="Hidden layer")(hidden_encoder_layer_1)
    latent_space = Dense(2, activation="logistic",
                         name="Latent space")(hidden_encoder_layer_2)
    encoder = Model(x, latent_space, name="Encoder")

    input_decoder = Input(shape=(2,), name="Decoder Input")
    hidden_decoder_layer_1 = Dense(
        10, activation="relu", name="Hidden layer")(input_decoder)
    hidden_decoder_layer_2 = Dense(
        20, activation="relu", name="Hidden layer")(hidden_decoder_layer_1)
    x_decoded = Dense(35, activation="logistic",
                      name="Output")(hidden_decoder_layer_2)
    decoder = Model(input_decoder, x_decoded, name="Decoder")

    # Create autoencoder
    autoencoder = Autoencoder(encoder, decoder, optimizer='powell')
    start = time.time()
    # Create dataset
    autoencoder.fit(raw_dataset, raw_dataset, epochs=10)
    end = time.time()
    print("Training time: ", end - start)
    # Test autoencoder
    prediction = autoencoder(raw_dataset[3])

    plot(prediction.reshape((7, 5)))

    # Plot latent space
    plot_latent_space(encoder, labelled_dataset)
