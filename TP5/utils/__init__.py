import matplotlib.pyplot as plt
import numpy as np

from models.Model import Model

monocromatic_cmap = plt.get_cmap('binary')


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

def plot_latent_space(encoder: Model, dataset: list[list[int]]):
    for i in range(len(dataset)):
        # 2D point
        predicted_point = encoder(dataset[i])
        plt.scatter(predicted_point[0], predicted_point[1])
    
    plt.show()
