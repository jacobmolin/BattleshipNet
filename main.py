from load_dataset import load_dataset

dataset_name = "batchsize_50_total_samples_500_date_20211029_2000"

data, labels = load_dataset(dataset_name)

print("data.shape:", data.shape)
print("lablels.shape:", labels.shape)
