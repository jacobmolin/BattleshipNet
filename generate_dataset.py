from utils import batch_data, givedata
import numpy as np
from datetime import datetime
"""
    Data will be in batches of 'batchsize' amount of independent 
    random boards where they go from non-explored to very explored.
"""

batchsize = 50  # How many boards to train on in each training session
reps = 10**2  # How many training sessions

data = []
labels = []

for r in range(reps):
    # print("r =", r)
    d, l = batch_data(batchsize=batchsize, autoencode=False)
    # print("d.shape:", d.shape)
    # print("l.shape:", l.shape)
    # data.extend(d)
    # labels.extend(l)
    data.append(d)
    labels.append(l)

data = np.asarray(data)
labels = np.asarray(labels)

print("data.shape:", data.shape)
print("lablels.shape:", labels.shape)

## Check the data
# for i in range(6):
#     idx = int(i * batchsize / 5)
#     if i == 5:
#         idx -= 1

#     print("============= data[{}] =============".format(idx))
#     # print(np.reshape(data[idx], [10, 10]))
#     unique, counts = np.unique(data[idx], return_counts=True)
#     d_u = dict(zip(unique, counts))
#     print("d_u =", d_u)
#     print("not 0.0 in d_u =", 100 - d_u[0.0])

time_format = datetime.now().strftime("%Y%m%d_%H%M")

np.save(
    "./datasets/" + "batchsize_" + str(batchsize) + "_total_samples_" +
    str(batchsize * reps) + "_date_" + time_format + "_data",
    data,
)

np.save(
    "./datasets/" + "batchsize_" + str(batchsize) + "_total_samples_" +
    str(batchsize * reps) + "_date_" + time_format + "_labels",
    labels,
)
