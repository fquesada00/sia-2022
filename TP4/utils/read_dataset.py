import pandas as pd


def read_dataset(path: str) -> pd.DataFrame:
    """
    Read the dataset from the given path.

    :param path: The path to the dataset.
    :return: The dataset.
    """
    return pd.read_csv(path, header=0)
