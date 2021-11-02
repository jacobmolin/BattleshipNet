from statistics import mean, median
from tensorflow.keras.models import load_model
from utils import png_to_gif, randboard, player, show_save_histogram, load_dataset
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


episodes = 1000
all_games_moves = []
model_type = "fully_connected"
total_samples = 5000
# model_name = model_type + "_bsz_50_totsampl_5000_f_7_e_20211030-1131"
# model_name = model_type + "_bsz_50_sampl_5000_epochs_29_20211031-1659"
# model_name = model_type + "_bsz_50_sampl_50000_epochs_50_20211031-1708"
# model_name = model_type + "_bsz_50_sampl_50000_epochs_42_20211031-1800"
model_name = (
    model_type + "_bsz_50_sampl_" + str(total_samples) + "_epochs_37_20211031-2133"
)
print("Model name:", model_name)
# model_path = (
#     "/Users/jacobmolin/Dropbox/LiU/mt5/HT1/TNM095/BattleshipNew/models/"
#     + model_type
#     + "_bsz_50_totsampl_5000_f_7_e_20211030-1131.h5"
# )
model_path = (
    "/Users/jacobmolin/Dropbox/LiU/mt5/HT1/TNM095/BattleshipNew/models/"
    + model_name
    + ".h5"
)
loaded_model = load_model(model_path)
loaded_model.summary()
show = False
save = False
save_histogram = True
show_histogram = False

# Run episodes number of games
for ep in range(episodes):
    board = randboard()
    # print(board)
    # gamename = model_name + "_" + datetime.now().strftime("%Y%m%d_%H%M")
    # print("gamename:", gamename)

    nr_of_moves = player(
        loaded_model,
        board,
        total_samples=total_samples,
        show=show,
        save=save,
        model_name=model_name,
        model_type=model_type,
    )

    if ep % int(episodes / 10) == 0:
        print("Episode {} ended with {} moves".format(ep, nr_of_moves))

    all_games_moves.append(nr_of_moves)

    # if save:
    #     png_to_gif()

# Print some data
median_moves = median(all_games_moves)
avg_moves = mean(all_games_moves)
best_game = min(all_games_moves)
worst_game = max(all_games_moves)
print("Avg moves for {} episodes = {}".format(episodes, avg_moves))
print("Median moves for {} episodes = {}".format(episodes, median_moves))
print("Best game: {} moves".format(best_game))
print("Worst game: {} moves".format(worst_game))


# ==================== Histogram of game moves ====================

print("Amount of games: {}".format(len(all_games_moves)))
print("Amount of moves each round:\n{}".format(all_games_moves))

show_save_histogram(
    all_games_moves,
    episodes,
    model_name,
    model_type,
    avg_moves,
    median_moves,
    show=show_histogram,
    save=save_histogram,
)
