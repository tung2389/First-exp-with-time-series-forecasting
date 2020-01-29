import numpy as np
import matplotlib.pyplot as plt

time = np.arange(4 * 365 + 1) 
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period 
    return amplitude * seasonal_pattern(season_time)


def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

trend_ = trend(time, slope)
season = seasonality(time, period=365, amplitude=amplitude)
noise = white_noise(time, noise_level, seed=42)

series = baseline + trend_ + season + noise