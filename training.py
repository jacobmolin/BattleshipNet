from load_dataset import load_dataset
import numpy as np
from math import floor
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses, metrics

# ==================== LOAD DATA ====================
print("tf.__version__:", tf.__version__)
batch_size = 50
total_samples = 500

# dataset_name = "batchsize_50_total_samples_150_date_20211029_2000"
# dataset_name = "batchsize_50_total_samples_500_date_20211030_0932"
dataset_name = "batchsize_" + str(batch_size) + "_total_samples_" + str(
    total_samples) + "_date_20211030_0932"
# dataset_name = "batchsize_" + str(batch_size) + "_total_samples_" + str(
#     total_samples) + "_date_20211030_0956"
# dataset_name = "batchsize_50_total_samples_50000_date_20211030_0955"

data, labels = load_dataset(dataset_name)

print("data.shape:", data.shape)
print("lablels.shape:", labels.shape)

# print(np.reshape(data[0], [10, 10]))
# print(np.reshape(data[1], [10, 10]))
# print(np.reshape(data[2], [10, 10]))

# print("======== LABELS ========")
# print(np.reshape(labels[0][0], [10, 10]))
# print(np.reshape(labels[1][0], [10, 10]))
# print(np.reshape(labels[2][0], [10, 10]))

# Reserve 15% data for validation and evaluation, respectively
val_eval_amount = floor(data.shape[0] * 0.30)
val_eval_data = data[-val_eval_amount:]
val_eval_labels = labels[-val_eval_amount:]
# val_data, eval_data = np.split(val_eval_data, 2)
# val_labels, eval_labels = np.split(val_eval_labels, 2)
val_amount = floor(val_eval_data.shape[0] * 0.50)
val_data, eval_data = val_eval_data[-val_amount:], val_eval_data[:-val_amount]
val_labels, eval_labels = val_eval_labels[
    -val_amount:], val_eval_labels[:-val_amount]

# Set training data to remaining 70%
data = data[:-val_eval_amount]
labels = labels[:-val_eval_amount]

print("data.shape:", data.shape)
print("labels.shape:", labels.shape)
print("val_data.shape:", val_data.shape)
print("val_labels.shape:", val_labels.shape)
print("eval_data.shape:", eval_data.shape)
print("eval_labels.shape:", eval_labels.shape)

# ==================== BUILD MODEL ====================

# Construct model
model_type = "fully_connected"
# model = build_cnn_model(model_type)

x_in = Input(shape=[None, 100], dtype=tf.float32)
x = Dense(100, activation="sigmoid")(x_in)
x = Dense(100, activation="sigmoid")(x)
x = Dense(100)(x)  # logit?

model = Model(inputs=x_in, outputs=x, name=type)

# Compile model
model.compile(
    # optimizer=Adam(learning_rate=0.001),
    optimizer=Adam(learning_rate=0.001,
                   beta_1=0.9,
                   beta_2=0.999,
                   epsilon=1e-08),
    # optimizer=SGD(learning_rate=0.01, momentum=0.9),
    # optimizer="adam",
    # Loss function to minimize
    loss=losses.BinaryCrossentropy(from_logits=True),
    # loss=losses.CategoricalCrossentropy(),
    # List of metrics to monitor
    # metrics=[metrics.Accuracy(dtype="float32")]
    metrics=[metrics.MeanAbsoluteError()]
    # metrics=[metrics.BinaryAccuracy()]
    # metrics=["accuracy"]
)

# Display model"s architecture
model.summary()

# ==================== TRAIN MODEL ====================

initial_epoch = 0
epochs = 1

# Training
history = model.fit(
    data,
    labels,
    # steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    initial_epoch=initial_epoch,
    # batch_size=batch_size,
    validation_data=(val_data, val_labels),
    # callbacks=[cp_callback] # Pass CP callback to training
)

# #  Save model to HDF5 file
# time_format = datetime.now().strftime("%Y%m%d-%H%M")
# filepath = "saved_models/" + model_type + "_bsz_" + str(
#     batch_size) + "_totsampl_" + str(total_samples) + "_f_" + str(
#         epochs) + "_e_" + time_format + ".h5"
# model.save(filepath)
# print("Saved model to {}".format(filepath))
