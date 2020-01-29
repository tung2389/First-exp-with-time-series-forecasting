import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.getcwd())
from utils.data import time, series, plot_series


split_time = 1000
time_train = time[:split_time] # From element 0 to element 999
value_train = series[:split_time]
time_valid = time[split_time:] # From element 1000 to 1460
value_valid = series[split_time:]

naive_forecast = series[split_time - 1:-1] 
print(naive_forecast)
# Copy the value from day 999 to 1459
# Naive forecast takes the value of the today and regard it as the predicted value for the next day
plt.figure(figsize=(10, 6))
plot_series(time_valid, value_valid, start=0, end=150, label="Series")
plot_series(time_valid, naive_forecast, start=0, end=151, label="Forecast")
plt.show()

def printTheLoss():
    errors = naive_forecast - value_valid
    abs_errors = np.abs(errors)
    loss = abs_errors.mean()
    print(loss)

printTheLoss()