import matplotlib.pyplot as plt
import numpy as np

from models.Model import Model

monocromatic_cmap = plt.get_cmap('binary')


def to_raw_dataset(labelled_dataset: list[dict]):
    raw_dataset = []

    for labelled_character in labelled_dataset:
        raw_dataset.append(labelled_character["bitmap"])

    return raw_dataset


def to_bin_array(encoded_caracter):
    bin_array = np.zeros((7, 5), dtype=int)
    for row in range(0, 7):
        current_row = encoded_caracter[row]
        for col in range(0, 5):
            bin_array[row][4-col] = current_row & 1
            current_row >>= 1
    return bin_array


def plot(character):
    plt.imshow(
        character,
        cmap=monocromatic_cmap)
    plt.show()


def plot_latent_space(encoder: Model, labelled_dataset: list[dict]):
    fig, ax = plt.subplots()

    latent_space = np.stack([encoder(bitmap.flatten()) for bitmap in map(
        to_bin_array, to_raw_dataset(labelled_dataset))])

    ax.scatter(latent_space[:, 0], latent_space[:, 1])
    ax.set_xlabel('$z_1$')
    ax.set_ylabel('$z_2$')

    for i, txt in enumerate(labelled_dataset):
        ax.annotate(txt["char"], (latent_space[i, 0], latent_space[i, 1]))

    plt.show()


def add_noise(image: np.ndarray, mode: str, amount: float = 0.1) -> np.ndarray:
    """
     Add noise to an image.

     Parameters
     ----------
     image : ndarray
         Input image data. Will be converted to float.
     mode : str
         One of the following strings, selecting the type of noise to add:

         'gauss'     Gaussian-distributed additive noise.

         'poisson'   Poisson-distributed noise generated from the data.
     amount : float
            Amount of noise to add. Default is 0.1. Must be a value between 0 and 1.

     """
    if mode == "gauss":
        row, col, ch = image.shape
        mean = 0
        sigma = 1
        gauss = np.random.normal(mean, sigma, (row, col, ch)) * amount
        # gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss

    elif mode == "s&p":
        row, col, ch = image.shape
        noisy = np.copy(image)
        s_vs_p = 0.5
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        noisy[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        noisy[coords] = 0

    # ensure values are between 0 and 1
    return np.clip(noisy, 0., 1.)
