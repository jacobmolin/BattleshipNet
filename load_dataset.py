import numpy as np


def load_dataset(
    dataset_name="",
    dataset_date="",
):

    data = np.load("./datasets/" + dataset_name + "_data.npy", allow_pickle=True)
    labels = np.load("./datasets/" + dataset_name + "_labels.npy", allow_pickle=True)

    return data, labels
