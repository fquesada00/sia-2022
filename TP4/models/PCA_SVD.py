import numpy as np
from sklearn.decomposition import PCA


class PCA_SVD:
    @classmethod
    def compute_pc1(self, input_dataset: np.ndarray) -> np.ndarray:
        pca = PCA()

        pca.fit(input_dataset)

        components = pca.components_

        return components[0]
