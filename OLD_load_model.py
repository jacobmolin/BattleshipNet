import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from statistics import median, mean
# from CNNBattleshipEnv import BattleshipEnv, CellAction
# from BattleshipModel import BattleshipModel

print("TF version:", tf.__version__)

# Load model
model_name = 'simple_800f_10e_20211020-2233'
loaded_model = tf.keras.models.load_model('saved_models/' + model_name + '.h5')
# # loaded_model = load_model('saved_models/' + model_name + '.h5', custom_objects={'customAccuracy': customAccuracy})
# # loaded_model.summary()

# ships = {}
# ships['carrier'] = 5
# ships['battleship'] = 4
# ships['cruiser'] = 3
# ships['submarine'] = 3
# ships['destroyer'] = 2
# grid_size = 10
# # Instatiate environment
# env = BattleshipEnv(grid_size=grid_size, ships=ships)

# # Run eval_eps number of games and calculate median and average amount of moves to finish game
# eval_eps = 20  # Number of games to play
# all_games_moves = []
# clips = []
# heatmaps = []
# enemy_boards = []
# all_taken_actions = []

# for i in range(eval_eps):
#     # Generate a random board
#     env.reset()
#     done = False
#     moves = 0
#     frames = []
#     heatmap = []
#     taken_actions = []

#     # Play a game (one epoch)
#     while not done and moves <= 100:
#         # Extract board
#         board = np.asarray(env.board.copy())

#         # Add board frame
#         frames.append(board[:, :, 0])

#         # Give model the current state of board and get prediction
#         predict = loaded_model.predict(np.array([board]))
#         heatmap.append(predict.reshape((grid_size, grid_size)))
#         predict[0, [
#             i for i, e in enumerate(
#                 np.reshape(env.board[:, :, 1], (grid_size * grid_size)))
#             if e == CellAction.Illegal.value
#         ]] = 0

#         # Extract location to shoot according to highest pobability in predict
#         action = np.unravel_index(
#             np.argmax(predict),
#             (grid_size, grid_size))  # 2D index of max value in predict
#         taken_actions.append(action)

#         # Attack
#         done = env.step(action=action)

#         moves += 1

#     all_games_moves.append(moves)
#     all_taken_actions.append(taken_actions)

#     # Add all frames to a new clip
#     clips.append(frames)
#     heatmaps.append(heatmap)
#     enemy_boards.append(env.enemy_board)

#     if not done:
#         print('Maximum iterations reached')  # Should never happen
#     print('Round {} of {} finished with {} moves'.format(
#         i + 1, eval_eps, moves))

# median_moves = median(all_games_moves)
# avg_moves = mean(all_games_moves)
# print('Median amount of moves until one round is completed: {}'.format(
#     median_moves))
# print('Average amount of moves until one round is completed: {}'.format(
#     avg_moves))
