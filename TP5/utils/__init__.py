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
