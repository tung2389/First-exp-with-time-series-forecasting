import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
from tensorflow import keras
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())
from utils.data import time, series, plot_series

def window_dataset(series, window_size, batch_size=32,
                   shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    # "This dataset fills a buffer with buffer_size elements, then randomly samples elements from this buffer"
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    # Prepare later batch while training current batch. This helps to improve performance at the cost
    # of using more memory
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# Set a fixed seed so we get the same result each time running the code
tf.random.set_seed(42)
np.random.seed(42)

window_size = 30
train_set = window_dataset(x_train, window_size)
valid_set = window_dataset(x_valid, window_size)

def getPlotToFindTheBestLearningRate(model):
    lr_schedule = keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-6 * 10**(epoch / 30)) # 10**x is 10^x. Therefore, the learning rate will be 
                                               # ten times bigger each 30 epochs
    optimizer = keras.optimizers.SGD(lr=1e-6, momentum=0.9)
    model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
    history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.axis([1e-6, 1e-3, 0, 20])
    plt.show()

linear_model = keras.models.Sequential([
    keras.layers.Dense(1, input_shape=[window_size])
])
dense_model = keras.models.Sequential([
    keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(1)
])
# getPlotToFindTheBestLearningRate(linear_model)
# getPlotToFindTheBestLearningRate(dense_model)

def train_linear_model():
    model = linear_model
    optimizer = keras.optimizers.SGD(lr=6*1e-5, momentum=0.9)
    model.compile(loss=keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])
    early_stopping = keras.callbacks.EarlyStopping(patience=10)
    model.fit(train_set, epochs=500, validation_data=valid_set, callbacks=[early_stopping])
    model.save(os.getcwd() + "/Machine learning forecast/model.h5")

def train_dense_model():
    model = dense_model
    optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)
    model.compile(loss=keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])
    early_stopping = keras.callbacks.EarlyStopping(patience=10)
    model.fit(train_set, epochs=500, validation_data=valid_set, callbacks=[early_stopping])
    model.save(os.getcwd() + "/Machine learning forecast/dense_model.h5")
    
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series) # Create a Dataset from series
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

train_dense_model()
# array[:, 0] takes only the [...][0] elements. in this case, because the forecast has shape (461,1),
# something like [[23]], we need to use [:, 0] to make it one dimension
# linear_forecast = model_forecast(keras.models.load_model(os.getcwd() + "/Machine learning forecast/model.h5"), series[split_time - window_size:-1], window_size)[:, 0]
# plt.figure(figsize=(10, 6))
# plot_series(time_valid, x_valid)
# plot_series(time_valid, linear_forecast)
# plt.show()
# print(keras.metrics.mean_absolute_error(x_valid, linear_forecast).numpy())