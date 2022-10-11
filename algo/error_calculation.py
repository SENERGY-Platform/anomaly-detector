import numpy as np
from sklearn.ensemble import IsolationForest
from statistics import median

def get_anomalous_indices(list_of_errors,contam):
    anomalous_indices = []
    anomalous_error_model = IsolationForest(contamination=contam).fit(np.array(list_of_errors).reshape(-1,1))
    predictions = anomalous_error_model.predict(np.array(list_of_errors).reshape(-1,1))
    for i in range(len(list_of_errors)):
        if predictions[i]==-1 and list_of_errors[i]>median(list_of_errors):
            anomalous_indices.append(i)
    return anomalous_indices

def get_anomalous_time_windows(data_series, anomalous_indices, window_length):
    anomalous_windows = []
    for index in anomalous_indices:
        anomalous_windows.append(data_series.iloc[index*window_length:index*window_length+window_length])
    return anomalous_windows

