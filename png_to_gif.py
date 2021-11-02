import os
from utils import png_to_gif

model_type = "fully_connected"
game_played_on = "20211031-2116"
dir = os.path.join("played_games", model_type + "_" + game_played_on)
print(dir)

model_name = "fully_connected_bsz_50_sampl_50000_epochs_42_20211031-1800"
file_names = "battleship_gameplay_" + model_name + "_" + game_played_on
print(file_names)


png_to_gif(dir, file_names, rate=4)
