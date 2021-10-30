import time
import random
import os
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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


def gameover(board, history):
    total = 0
    for p in history:
        total += board[p[0]][p[1]]
    if total < (2 + 3 + 3 + 4 + 5):
        return False
    else:
        return True


def showit(board, history, show, save, gamename):
    board = board - 0.5
    for p in history:
        board[p[0]][p[1]] *= 2
    plt.imshow(board, interpolation='nearest', cmap='Greys')

    if save:
        dir = "played_games/" + gamename + "/game"
        isExist = os.path.exists(dir)

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(dir)

        file_name = "battleship_gameplay_" + gamename + "_" + str(
            len(history)) + ".png"

        file_path = os.path.join(dir, file_name)
        # print("file_path:", file_path)
        plt.savefig(file_path, bbox_inches='tight')

    if show:
        plt.show()


def showit_probs(probs, history, show, save, gamename):
    # probs = probs - 0.5
    # for p in history:
    #     probs[p[0]][p[1]] *= 2
    plt.imshow(probs, interpolation='nearest', cmap='Greys')

    if save:
        dir = "played_games/" + gamename + "/probs"
        isExist = os.path.exists(dir)

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(dir)

        file_name = "battleship_probs_" + gamename + "_" + str(
            len(history)) + ".png"

        file_path = os.path.join(dir, file_name)
        # print("file_path:", file_path)
        plt.savefig(file_path, bbox_inches='tight')

    if show:
        plt.show()


def makemove(model, board, history):
    #board is the full board
    #history is a list of positions which have already been attacked
    #eg: [(0, 1), (5, 3), (9, 0)]
    if len(history) == 100:
        return None

    knowledge = np.zeros([10, 10])
    knowledge2 = np.zeros([10, 10])
    for pos in history:
        knowledge[pos[0]][pos[1]] = board[pos[0]][pos[
            1]] * 2 - 1  # Set value at know positions to Hit = 1 or Miss = -1
        knowledge2[pos[0]][pos[1]] = board[pos[0]][pos[1]]

    # print("========= knowledge =========")
    # print(knowledge)
    # print("========= knowledge2 =========")
    # print(knowledge2)

    knowledge = knowledge.reshape((1, -1))

    probs = model.predict(knowledge)

    # TODO CHECK IF THIS IS NEEDED IF WE TRAIN WITH SIGMOID IN
    # THE FINAL LAYER LIKE WITH sigmoid_cross_entropy_with_logits
    probs = np.reshape(sigmoid(probs), [10, 10])

    # TODO Show/save probability distribution for a gameplay
    # show_probs(probs, history, show=True, save=False, gamename=gamename)

    choices = []
    for i, v in np.ndenumerate(probs):
        choices.append((v, i))
    choices.sort(key=lambda x: x[0], reverse=True)

    # print("========= CHOICES =========")
    # print(choices)
    # print("========= HISTORY =========")
    # print(history)

    for c in choices:
        if c[1] not in history:
            return c[1], probs


def player(model,
           board,
           show=True,
           save=False,
           show_probs=False,
           save_probs=False,
           gamename=""):
    #plays a game and returns how  many moves it took

    history = []

    counter = 0

    while not gameover(board, history):
        move, probs = makemove(model, board, history)
        history.append(move)

        # if counter >= 2:
        #     break
        # counter += 1

        if show or save:
            showit(board, history, show, save, gamename)

        if show_probs or save_probs:
            showit_probs(probs, history, show_probs, save_probs, gamename)

    return len(history)
