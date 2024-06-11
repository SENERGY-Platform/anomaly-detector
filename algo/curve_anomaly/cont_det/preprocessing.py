import numpy as np
from scipy.signal import savgol_filter

def normalize_data(data_series, training_max):
    return data_series/training_max

def minute_resampling(data_series):
    resampled_data_series = data_series.resample('s').interpolate().resample('T').asfreq().dropna()
    return resampled_data_series

def smooth_data(data_series):
    data_series_hat = data_series.rolling(window=30, win_type="exponential", center=True).mean().fillna(value=0)
    return data_series_hat

def decompose_into_time_windows(data_series, window_length=405):
    data_matrix = []
    for i in range(len(data_series), 0, -window_length):
        if i-window_length >= 0:
            data_matrix.append(data_series[i-window_length:i])
    data_matrix.reverse()
    return np.array(data_matrix)
