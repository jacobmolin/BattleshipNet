from readDataset import readDataset
from math import floor
import numpy as np

# height x length x value (hit, miss, unknown)
#   10   x   10   x   3

# Load data
amount_train_formations = 800 # Here this is just to choose the dataset
train_boards, train_labels = readDataset(amount_train_formations, shuffled=True)
print('Data loaded!')

# erve 15% data for validation and evaluation, respectively
val_eval_amount = floor(train_boards.shape[0] * 0.30)
val_eval_boards = train_boards[-val_eval_amount:]
val_eval_labels = train_labels[-val_eval_amount:]
# val_boards, eval_boards = np.split(val_eval_boards, 2)
# val_labels, eval_labels = np.split(val_eval_labels, 2)
val_amount = floor(val_eval_boards.shape[0] * 0.50)
val_boards, eval_boards = val_eval_boards[-val_amount:], val_eval_boards[:-val_amount]
val_labels, eval_labels =  val_eval_labels[-val_amount:], val_eval_labels[:-val_amount]

# Set training data to remaining 70%
train_boards = train_boards[:-val_eval_amount]
train_labels = train_labels[:-val_eval_amount]

print(train_boards.shape)
print(train_labels.shape)
print(val_boards.shape)
print(val_labels.shape)
print(eval_boards.shape)
print(eval_labels.shape)