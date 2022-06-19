import time
from matplotlib import pyplot as plt
import numpy as np
from models.Autoencoder import Autoencoder, create_autoencoder
from utils import to_bin_array, to_raw_dataset
from models.Model import Model
from utils import prepare_plot_5n_unlabelled_letters
from utils.dataset import font_1, font_2, font_3


def get_direction_vector(source, target):
    return (target[0] - source[0], target[1] - source[1])


def get_distance(source, target):
    return ((target[0] - source[0]) ** 2 + (target[1] - source[1]) ** 2) ** 0.5


def normalize_vector(vector, distance):
    return (vector[0] / distance, vector[1] / distance)


def move_point_in_direction(point, direction, distance):
    return (point[0] + direction[0] * distance, point[1] + direction[1] * distance)


def get_n_points_in_line(source, target, n):
    """
    Returns a list of n points in the line from source to target including source & target.
    """
    direction = get_direction_vector(source, target)
    distance = get_distance(source, target)
    normalized_direction = normalize_vector(direction, distance)
    step_size = distance / (n + 1)
    points = []
    for i in range(n+2):
        points.append(move_point_in_direction(
            source, normalized_direction, step_size * i))
    return points


def generate_n_new_letters(decoder: Model, source, target, n: int = 1):
    """
    Decode to generate n new letters between source & target.
    """
    points = get_n_points_in_line(source, target, n)
    decoded_letters = []
    for i in range(n+2):
        point = (points[i][0].round(2), points[i][1].round(2))
        decoded_letters.append({
            "letter": decoder(np.array([point[0], point[1]])).reshape((7, 5)),
            "point": point
        })
    return decoded_letters


def plot_generated_letters(generated_letters):
    prepare_plot_5n_unlabelled_letters(generated_letters, n=len(generated_letters))
    plt.show()
    

def create_and_train_default_autoencoder(font):
    labelled_dataset = font

    raw_dataset = np.array([data.flatten() for data in map(
        to_bin_array, to_raw_dataset(labelled_dataset))])
    
    raw_dataset = raw_dataset[:5]

    # Create autoencoder
    input_size = len(raw_dataset[0])
    latent_space_size = 2
    encoder_layers = [25, 10, latent_space_size]
    decoder_layers = [10, 25, input_size]
    encoder_activations = ["relu", "logistic", "relu"]
    decoder_activations = ["relu", "logistic", "logistic"]

    encoder, decoder = create_autoencoder(
        input_size, encoder_layers, decoder_layers, encoder_activations, decoder_activations)

    autoencoder = Autoencoder(encoder, decoder, optimizer='powell')

    start = time.time()
    autoencoder.fit(raw_dataset, raw_dataset, epochs=2)
    end = time.time()

    print("Training time: ", end - start)

    return encoder, decoder, autoencoder

def main():
    font = font_2
    encoder, decoder, autoencoder = create_and_train_default_autoencoder(font)
    source = encoder(to_bin_array(font[1]["bitmap"]).flatten())
    target = encoder(to_bin_array(font[2]["bitmap"]).flatten())
    generated_letters = generate_n_new_letters(decoder, source, target, n=10)
    plot_generated_letters(generated_letters)


if __name__ == "__main__":
    main()