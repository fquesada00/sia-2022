from pprint import pprint
import numpy as np

from models import Kohonen  
from utils.read_dataset import read_dataset


def main():
    data = read_dataset('./datasets/europe.csv')

    # remove Country column from dataset
    data_no_countries = data.drop(['Country'], axis=1)

    # scale dataset
    data_scaled = (data_no_countries - data_no_countries.mean()
                   ) / data_no_countries.std()
    k = 7
    kohonen = Kohonen(k, data_scaled,10,0.9)
    elems = kohonen.train(data_scaled, 100)
    neuron_map = kohonen.test(data_scaled)
    pprint(neuron_map)

if __name__ == '__main__':
    main()
