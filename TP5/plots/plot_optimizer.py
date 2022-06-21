from cProfile import label
import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np


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

        if file_name.startswith("error") and (file_name.endswith('powell.txt') or file_name.endswith('cg.txt') or file_name.endswith('bfgs.txt')):
            errors_files[file_name[file_name.rindex(
                "_") + 1:file_name.index('.')]] = os.path.join('data', file_name)

    pprint(errors_files)
    plot_error(errors_files)
