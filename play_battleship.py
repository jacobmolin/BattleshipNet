from re import S
from statistics import mean, median
from tensorflow.keras.models import load_model
from utils import randboard, player
from load_dataset import load_dataset
import numpy as np
from datetime import datetime


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


episodes = 1
all_games_moves = []
model_type = "fully_connected"

model_path = "/Users/jacobmolin/Dropbox/LiU/mt5/HT1/TNM095/BattleshipNew/saved_models/" + model_type + "_bsz_50_totsampl_5000_f_7_e_20211030-1131.h5"
loaded_model = load_model(model_path)
loaded_model.summary()

for ep in range(episodes):

    board = randboard()
    # print(board)
    gamename = model_type + "_" + datetime.now().strftime("%Y%m%d_%H%M")
    print("gamename:", gamename)

    nr_of_moves = player(loaded_model,
                         board,
                         show=False,
                         save=True,
                         show_probs=False,
                         save_probs=True,
                         gamename=gamename)
    print("Episode {} ended with {} moves".format(ep, nr_of_moves))
    all_games_moves.append(nr_of_moves)

median_moves = median(all_games_moves)
avg_moves = mean(all_games_moves)
best_game = min(all_games_moves)
worst_game = max(all_games_moves)
print("Avg moves for {} episodes = {}".format(episodes, avg_moves))
print("Median moves for {} episodes = {}".format(episodes, median_moves))
print("Best game: {} moves".format(best_game))
print("Worst game: {} moves".format(worst_game))