import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import sys, os
sys.path.append(os.getcwd())
from utils.data import time, series, plot_series
from utils.prepDataset import create_window_dataset, create_seq2seq_window_dataset
from utils.modelForecast import model_forecast


split_time = 1000
time_train = time[:split_time] # From element 0 to element 999
value_train = series[:split_time]
time_valid = time[split_time:] # From element 1000 to 1460
value_valid = series[split_time:]

tf.random.set_seed(42)
np.random.seed(42)

window_size = 30
# Shape of a batch [[1,2,3, ...], [...]]
train_set = create_window_dataset(value_train, window_size, batch_size=128)
valid_set = create_window_dataset(value_valid, window_size, batch_size=128)

seq_train_set = create_seq2seq_window_dataset(value_train, window_size,
                                   batch_size=128)
seq_valid_set = create_seq2seq_window_dataset(value_valid, window_size,
                                   batch_size=128)

# for X_batch, Y_batch in create_window_dataset(tf.range(10), 3,
#                                                batch_size=1):
#     print("X:", X_batch.numpy())
#     print("Y:", Y_batch.numpy())
# Output: X: [[4 5 6]]
#         Y: [7]

def getPlotToFindTheBestLearningRate(model, train_set):
  lr_schedule = keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-7 * 10**(epoch / 20))
  optimizer = keras.optimizers.SGD(lr=1e-7, momentum=0.9)
  model.compile(loss=keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])
  history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
  plt.semilogx(history.history["lr"], history.history["loss"])
  plt.axis([1e-7, 1e-4, 0, 30])
  plt.show()

simple_model = keras.models.Sequential([
  keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), # Add another dimension to the input
                      input_shape=[None]),                  # Now a batch has shape [ [ [1], [2], ... ] ]
  keras.layers.SimpleRNN(100, return_sequences=True),
  keras.layers.SimpleRNN(100),
  keras.layers.Dense(1),
  keras.layers.Lambda(lambda x: x * 200.0)  # Because the value of labels ranging from 40 -> 120 and the
                                            # output of the last Dense layer is between -1 and 1, we just
                                            # scale the final output so that the network can easily
                                            # compare it with the labels. You can choose some large number
                                            # like 120, 180, or 220 but here 200 is a good choice
                                            # as it leads to more accurate model
])

# getPlotToFindTheBestLearningRate(simple_model, train_set)

# Making prediction and caculate the loss at every time step (sequence to sequence model) helps to improve
# training speed
seqToseqModel = keras.models.Sequential([
  keras.layers.SimpleRNN(100, return_sequences=True,
                        input_shape=[None, 1]),
  keras.layers.SimpleRNN(100, return_sequences=True),
  keras.layers.Dense(1),
  keras.layers.Lambda(lambda x: x * 200.0)
])

# getPlotToFindTheBestLearningRate(seqToseqModel, seq_train_set)

def trainSimpleModel(): 
  optimizer = keras.optimizers.SGD(lr=1e-6, momentum=0.9)
  model = simple_model
  model.compile(loss=keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])
  early_stopping = keras.callbacks.EarlyStopping(patience=50)
  model_checkpoint = keras.callbacks.ModelCheckpoint(
      os.getcwd() + "/RNN forecast/checkpoint.h5", save_best_only=True)
  model.fit(train_set, epochs=500,
            validation_data=valid_set,
            callbacks=[early_stopping, model_checkpoint])
  rnn_forecast = model_forecast(
      model,
      series[split_time - window_size:-1],
      window_size)[:, 0]
  plt.figure(figsize=(10, 6))
  plot_series(time_valid, value_valid)
  plot_series(time_valid, rnn_forecast)
  plt.show()
  print(keras.metrics.mean_absolute_error(value_valid, rnn_forecast).numpy())

  
# for X_batch, Y_batch in seq2seq_window_dataset(tf.range(10), 3,
#                                                batch_size=1):
#     print("X:", X_batch.numpy())
#     print("Y:", Y_batch.numpy())

def trainSeqToSeqModel():
  optimizer = keras.optimizers.SGD(lr=5*1e-6, momentum=0.9)
  model = seqToseqModel
  model.compile(loss=keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])
  early_stopping = keras.callbacks.EarlyStopping(patience=10)
  model_checkpoint = keras.callbacks.ModelCheckpoint(
      os.getcwd() + "/RNN forecast/seq_checkpoint.h5", save_best_only=True)
  model.fit(seq_train_set, epochs=500,
            validation_data=seq_valid_set,
            callbacks=[early_stopping, model_checkpoint])
  # Series[..., np.newaxis] add one more dimension to series. For example: [1,2,3,4] -> [[1],[2],[3],[4]]
  # Series[..., np.newaxis] is equivalent to tf.expand_dims(series, axis=-1)
  rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size) 
  rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
  plt.figure(figsize=(10, 6))
  plot_series(time_valid, value_valid)
  plot_series(time_valid, rnn_forecast)
  plt.show()
  print(keras.metrics.mean_absolute_error(value_valid, rnn_forecast).numpy())

trainSeqToSeqModel()