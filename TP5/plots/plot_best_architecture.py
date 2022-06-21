from cProfile import label
import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np


def plot_error():

    plt.figure(figsize=(10, 10))

    error = np.loadtxt("data/error_25-10_2_200.txt")
    label = "25-10"
    plt.tight_layout()
    plt.xlabel("Época", fontsize=20)
    plt.ylabel("Error", fontsize=20)
    # Log scale
    plt.yscale("log")
    # Set fonts to 15
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.plot(error, label=label)

    plt.legend(fontsize=15)
    plt.show()


if __name__ == "__main__":

    plot_error()
