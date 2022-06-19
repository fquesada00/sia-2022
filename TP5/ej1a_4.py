from matplotlib import pyplot as plt
from main import create_and_train_default_autoencoder
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


def generate_new_letters(decoder: Model, source, target, n: int = 1):
    """
    Decode to generate n new letters between source & target.
    """
    points = get_n_points_in_line(source, target, n)
    decoded_letters = []
    for i in range(n):
        decoded_letters.append(decoder(points[i][0], points[i][1]))
    return decoded_letters


def plot_generated_letters(generated_letters):
    prepare_plot_5n_unlabelled_letters(generated_letters, n=len(generated_letters))
    plt.show()
    

def main():
    font = font_2
    encoder, decoder, autoencoder = create_and_train_default_autoencoder(font)
    source = encoder(font[1]["bitmap"])
    target = encoder(font[2]["bitmap"])
    generated_letters = generate_new_letters(decoder, source, target, n=3)
    plot_generated_letters(generated_letters)


if __name__ == "__main__":
    main()