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


def plot_5n_letters(output: np.ndarray, labelled_dataset: list[dict], n: int = 1):
    fig, axs = plt.subplots(
        n, 5, sharey=False, tight_layout=True, figsize=(12, 6), facecolor='white')

    for i in range(0, n):
        for j in range(0, 5):
            if (n > 1):
                ax = axs[i][j]
            else:
                ax = axs[j]

            ax.imshow(output[i * 5 + j], cmap=monocromatic_cmap)
            ax.set(title=labelled_dataset[i * 5 + j]["char"])

def prepare_plot_5n_unlabelled_letters(output, n: int = 1):
    fig, axs = plt.subplots(
        1, n, sharey=False, tight_layout=True, figsize=(12, 6), facecolor='white')
    for i in range(n):
        axs[i].imshow(output[i]["letter"], cmap=monocromatic_cmap)
        axs[i].set(title=f"{output[i]['point']}")


def plot_denoiser(test_set: np.ndarray, denoised_output: np.ndarray, labelled_dataset: list[dict]):
    fig, axs = plt.subplots(
        5, 3, sharey=False, tight_layout=True, figsize=(12, 6), facecolor='white')

    original_dataset = to_raw_dataset(labelled_dataset)

    for i in range(0, 5):
        axs[i][0].imshow(test_set[i].reshape(7, 5), cmap=monocromatic_cmap)
        axs[i][1].imshow(original_dataset[i].reshape(
            7, 5), cmap=monocromatic_cmap)
        axs[i][2].imshow(denoised_output[i].reshape(
            7, 5), cmap=monocromatic_cmap)
        axs[i][0].set(title=f"{labelled_dataset[i]['char']} - Entrada")
        axs[i][1].set(title=f"{labelled_dataset[i]['char']} - Deseado")
        axs[i][2].set(title=f"{labelled_dataset[i]['char']} - Resultado")

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

         's&p'   Salt and pepper noise.
     amount : float
            Amount of noise to add. Default is 0.1. Must be a value between 0 and 1.

     """
    if mode == "gauss":
        mean = 0
        sigma = 1
        gauss = np.random.normal(mean, sigma, image.shape) * amount
        # gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss

    elif mode == "s&p":
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
