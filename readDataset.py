import numpy as np

def readDataset(amount_formations, grid_size = 10, prefix = "./datasets/", shuffled = False):
    data = np.load("./datasets/" + str(grid_size) + "x" + str(grid_size) + "_" + str(amount_formations) + ('_shuffled' if shuffled else '') + '_data.npy', allow_pickle=True)
    labels = np.load("./datasets/" + str(grid_size) + "x" + str(grid_size) + "_" + str(amount_formations) + ('_shuffled' if shuffled else '') + '_labels.npy', allow_pickle=True)

    return data, labels
