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

    print(latent_space)

    ax.scatter(latent_space[:, 0], latent_space[:, 1])
    ax.set_xlabel('$z_1$')
    ax.set_ylabel('$z_2$')

    for i, txt in enumerate(labelled_dataset):
        ax.annotate(txt["char"], (latent_space[i, 0], latent_space[i, 1]))

    plt.show()


def add_noise(image: np.ndarray, mode: str, s_p_amount: float = 0.1, s_p_ratio: float = 0.5, gauss_var: float = 0.1):
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
     s_p_amount : float
            Amount of noise to add for salt and pepper noise.
     s_p_ratio : float
            Salt to pepper ratio salt/pepper.
     gauss_var : float
            Variance of the gaussian noise.
     """

    if mode == "gauss":
        row, col, ch = image.shape
        mean = 0
        sigma = gauss_var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif mode == "s&p":
        row, col, ch = image.shape
        noisy = np.copy(image)
        # Salt mode
        num_salt = np.ceil(s_p_amount * image.size * s_p_ratio)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        noisy[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(s_p_ratio * image.size * (1. - s_p_ratio))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        noisy[coords] = 0
        return noisy
