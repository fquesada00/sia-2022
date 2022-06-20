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


def plot_latent_space(encoder: Model, labelled_dataset: list[dict]):
    fig, ax = plt.subplots()

    latent_space = np.stack([encoder(bitmap.flatten()) for bitmap in map(
        to_bin_array, to_raw_dataset(labelled_dataset))])

    ax.scatter(latent_space[:, 0], latent_space[:, 1])
    ax.set_xlabel('$z_1$')
    ax.set_ylabel('$z_2$')

    for i, txt in enumerate(labelled_dataset):
        ax.annotate(txt["char"], (latent_space[i, 0], latent_space[i, 1]))

    plt.grid()
    plt.show()


def plot_decoded_latent_space(image_shape: tuple, decoder, grid_transform, digits: int = 10):
    figure = np.zeros((image_shape[0] * digits, image_shape[1] * digits))
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = grid_transform(np.linspace(0.05, 0.95, digits))
    grid_y = grid_transform(np.linspace(0.05, 0.95, digits))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(image_shape)
            figure[i * image_shape[1]: (i + 1) * image_shape[1],
                   j * image_shape[0]: (j + 1) * image_shape[0]] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
