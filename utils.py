import time
import random
import numpy as np


def validplacement(i, j, horz, board, boat):
    if horz:
        for k in range(boat):
            if i + k > 9:  # out of range
                return False
            if board[i + k][j] == 1:  # theres already a boat there
                return False
        return True
    else:
        for k in range(boat):
            if j + k > 9:  # out of range
                return False
            if board[i][j + k] == 1:  # theres already a boat there
                return False
        return True


def place(i, j, horz, board, boat):
    if horz:
        for k in range(boat):
            board[i + k][j] = 1
    else:
        for k in range(boat):
            board[i][j + k] = 1
    return board


def randboard():
    t1 = time.time()
    board = np.zeros([10, 10])
    boats = [2, 3, 3, 4, 5]
    np.random.shuffle(boats)
    while boats:
        # print('this')
        boat = boats[-1]
        horizontal = random.choice([True, False])
        badchoice = True
        while badchoice:
            # print('that')
            if horizontal:
                i = random.randrange(10 - boat + 1)
                j = random.randrange(10)
            else:
                j = random.randrange(10 - boat + 1)
                i = random.randrange(10)
            badchoice = not validplacement(i, j, horizontal, board, boat)
        board = place(i, j, horizontal, board, boat)
        boats.pop()
    t2 = time.time()
    # print(t2-t1)
    return board


def givedata(n, autoencode=False):
    # print("n = {}".format(n))
    label = randboard()
    if autoencode:
        data = label * 2 - 1
        return data.flatten(), label.flatten()

    xs = np.random.choice(10, n)  # n random elements in range [0..9]
    ys = np.random.choice(10, n)
    # print("xs.shape = {}, ys.shape = {}".format(xs.shape, ys.shape))
    # print("xs = {}, ys = {}".format(xs, ys))

    data = np.zeros([10, 10])

    for c in range(n):
        data[xs[c]][ys[c]] = label[xs[c]][ys[c]] * 2 - 1
        # print("c = {}".format(c))
        # print("xs[c] = {}, ys[c] = {}".format(xs[c], ys[c]))
        # print("label[xs[c]][ys[c]] = {}".format(label[xs[c]][ys[c]]))
        # print("data[xs[c]][ys[c]] = {}".format(data[xs[c]][ys[c]]))

    # print("===== DATA =====")
    # print(data)

    return data.flatten(), label.flatten()


def batch_data(batchsize, autoencode=False):
    ds = []
    ls = []
    # ds = np.ndarray()
    # ls = np.ndarray()
    for i in range(batchsize):
        # print("i = {}, batchsize = {}".format(i, batchsize))
        # print("int(100.0 * i / batchsize) = {}".format(int(100.0 * i / batchsize)))
        d, l = givedata(int(100.0 * i / batchsize), autoencode=autoencode)
        ds.append(d)
        ls.append(l)
    return np.asarray(ds), np.asarray(ls)
