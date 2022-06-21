from cProfile import label
import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np


def findnth(haystack, needle, n):
    parts = haystack.split(needle, n+1)
    if len(parts) <= n+1:
        return -1
    return len(haystack)-len(parts[-1])-len(needle)


def plot_error(error_files):

    plt.figure(figsize=(10, 10))

    errors = [np.loadtxt(error_files[error_file])
              for error_file in error_files]
    labels = [file_name for file_name in error_files]
    plt.tight_layout()
    plt.xlabel("Época", fontsize=20)
    plt.ylabel("Error", fontsize=20)
    # Log scale
    # plt.yscale("log")
    # Set fonts to 15
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    for index, error in enumerate(errors):
        plt.plot(error, label=labels[index])

    plt.legend(fontsize=15)
    plt.show()


if __name__ == "__main__":
    errors_files = {}
    for file_name in sorted(os.listdir('data')):

        if file_name.startswith('error') and file_name.endswith("100.txt"):
            latent_space_index = findnth(file_name, "_", 1)+1
            latent_space_end = findnth(file_name, "_", 2)
            errors_files[file_name[
                latent_space_index:latent_space_end
            ]] = os.path.join('data', file_name)

    # pprint(errors_files)
    plot_error(errors_files)
