import numpy as np
from models.PCA_SVD import PCA_SVD
from models.Oja import Oja
from utils.read_dataset import read_dataset


if __name__ == '__main__':
    data = read_dataset('./datasets/europe.csv')

    # remove Country column from dataset
    data_no_countries = data.drop(['Country'], axis=1)

    # scale dataset
    data_scaled = (data_no_countries - data_no_countries.mean()
                   ) / data_no_countries.std()

    # compute the first principal component with Oja's algorithm
    pc1_oja = Oja.compute_pc1(data_scaled.to_numpy(),
                              epochs=100000, learning_rate=0.0001)

    # compute the first principal component with SVD
    pc1_svd = PCA_SVD.compute_pc1(data_scaled.to_numpy())

    # compare
    print("With Oja's algorithm:")
    print(pc1_oja)

    print("With SVD:")
    print(pc1_svd)

    # distance
    print("Distance:")
    print(np.linalg.norm(pc1_oja - pc1_svd))
