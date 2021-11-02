from utils import load_dataset

batch_size = 50
total_samples = 50000

dataset_name = (
    "batchsize_"
    + str(batch_size)
    + "_total_samples_"
    + str(total_samples)
    + "_date_20211030_1323"
)


data, labels = load_dataset(dataset_name)

print("data.shape:", data.shape)
print("lablels.shape:", labels.shape)
