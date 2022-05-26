from pprint import pprint
import numpy as np

from .models import Kohonen


def main():
    print("Ejercicio 1a")
    dataset_input = np.array([[0, 1, 2, 3, 4, 5, 6],
                     [1, 2, 3, 4, 5, 6, 7],
                     [2, 3, 4, 5, 6, 7, 8],
                     [3, 4, 5, 6, 7, 8, 9],
                     [4, 5, 6, 7, 8, 9, 10],
                     [5, 6, 7, 8, 9, 10, 11],
                     [6, 7, 8, 9, 10, 11, 12],
                     [7, 8, 9, 10, 11, 12, 13],
                     [8, 9, 10, 11, 12, 13, 14],
                     [9, 10, 11, 12, 13, 14, 15],
                     [10, 11, 12, 13, 14, 15, 16],
                     [11, 12, 13, 14, 15, 16, 17],
                     [12, 13, 14, 15, 16, 17, 18],
                     [13, 14, 15, 16, 17, 18, 19],
                     [14, 15, 16, 17, 18, 19, 20],
                     [15, 16, 17, 18, 19, 20, 21]])
    k = 7
    kohonen = Kohonen(k, dataset_input,10,0.9)
    elems = kohonen.train(dataset_input, 100)
    neuron_map = kohonen.test(dataset_input)
    pprint(neuron_map)

if __name__ == '__main__':
    main()
