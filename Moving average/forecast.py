import numpy as np
import matplotlib.pyplot as plt
import os, sys
from tensorflow import keras
sys.path.append(os.getcwd())
from utils.data import time, series, plot_series

split_time = 1000
time_train = time[:split_time] # From element 0 to element 999
value_train = series[:split_time]
time_valid = time[split_time:] # From element 1000 to 1460
value_valid = series[split_time:]

def moving_average_forecast(series, window_size):
    """Forecasts the mean of the last few values.
        If window_size=1, then this is equivalent to naive forecast"""
    forecast = []
    for time in range(len(series) - window_size): # From 0 to len(series) - 1 - window_size
        forecast.append(series[time : time + window_size].mean()) # Append the mean from a specific
    return np.array(forecast)                                     # time to (that time + window_size)

def moving_average_forecast(series, window_size):
    """This implementation is *much* faster than the previous one"""
    mov = np.cumsum(series) # Return an array in which each element is the sum of all elements from 0 
                            # to that element.              
    # array[:-x] means that every elements from 0 to [len - 1 - x]
    mov[window_size:] = mov[window_size:] - mov[:-window_size] # The ith element holding sum from
                                                               # [i - window_size + 1] element to [i]
    return mov[window_size - 1:-1] / window_size # Caculate the mean values of a window_size interval 
                                                 # The array holding mean values starting from window_size - 1
                                                 # to the [len - 2] element (the second-last element)

moving_avg = moving_average_forecast(series, 30)[split_time - 30:] # 29 first elements in moving_average_forecast 
                                                                   # was cut off (because of return mov(window_size - 1))
                                                                   # Therefore, we need to add [split_time - 30:]

# plt.figure(figsize=(10, 6))
# plot_series(time_valid, value_valid, label="Series")
# plot_series(time_valid, moving_avg, label="Moving average (30 days)")
# plt.show()
# print(keras.metrics.mean_absolute_error(value_valid, moving_avg).numpy())

#Eliminate trend and seasonality
diff_series = (series[365:] - series[:-365]) # The diff_series has 1461 - 365 = 1096 elements
diff_time = time[365:]

# plt.figure(figsize=(10, 6))
# plot_series(diff_time, diff_series, label="Series(t) – Series(t–365)")
# plt.show()

diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:] #The array has been cut off 365 and then 50 elements

# plt.figure(figsize=(10, 6))
# plot_series(time_valid, diff_series[split_time - 365:], label="Series(t) – Series(t–365)")
# plot_series(time_valid, diff_moving_avg, label="Moving Average of Diff")
# plt.show()

# Predict: Series(t) = average_of_diff + series(t - 365)
# Series[split_time - 365] is series(t-365)
diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg

# plt.figure(figsize=(10, 6))
# plot_series(time_valid, value_valid, label="Series")
# plot_series(time_valid, diff_moving_avg_plus_past, label="Forecasts")
# plt.show()

# print(keras.metrics.mean_absolute_error(value_valid, diff_moving_avg_plus_past).numpy())

# We can use moving_average_forecast in any interval with length equal to 461 (len of diff_moving_avg). We can use
# the moving average of day 1000 - 365 to 1460 - 365 as:
# moving_average_forecast(series[split_time - 376:-365], 11); But in this case, use moving average
# from element 641th to 1101th is more effective 
# Ex:
# diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-359], 11) + diff_moving_avg

def calc_moving_avg_past_on_interval_461(start, window_size):
    return moving_average_forecast(series[start - window_size : -( 1460 - (start + 461) + 1)], window_size)

# diff_moving_avg plus moving average of past value from day 641 to day 1101
diff_moving_avg_plus_smooth_past = calc_moving_avg_past_on_interval_461(641, 11) + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, value_valid, label="Series")
plot_series(time_valid, diff_moving_avg_plus_smooth_past, label="Forecasts")
plt.show()

print(keras.metrics.mean_absolute_error(value_valid, diff_moving_avg_plus_smooth_past).numpy())