from pprint import pprint
import numpy as np

from models.Kohonen import Kohonen
from utils.read_dataset import read_dataset


def main():
    data = read_dataset('./datasets/europe.csv')

    # remove Country column from dataset
    data_no_countries = data.drop(['Country'], axis=1)

    # scale dataset
    data_scaled = (data_no_countries - data_no_countries.mean()
                   ) / data_no_countries.std()
    data_scaled_numpy = data_scaled.to_numpy()

    k = 4

    kohonen = Kohonen(k, data_scaled_numpy, 4, 0.01)
    # pprint(data_scaled_numpy)
    pprint(kohonen.weights)
    winner_idx_arr_row, winner_idx_arr_col, radius_arr, learning_rate_arr, dist_arr = kohonen.train(
        data_scaled_numpy, 1000)

    pprint(kohonen.weights)
    neuron_map = kohonen.test(data_scaled_numpy)
    pprint(neuron_map)
    # pprint(kohonen.weights)
    # pprint(list(zip(winner_idx_arr_row, winner_idx_arr_col)))


if __name__ == '__main__':
    main()
