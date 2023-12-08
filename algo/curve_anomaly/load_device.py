import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle

from . import preprocessing

def extract_loads(time_series):
    list_of_loads = []
    list_of_load_inds = []
    new_load = []
    end_check = []
    active = False
    for i in range(len(time_series)):
        if active == True:
            new_load.append(i)
            if time_series[i] < 1.5: # If power values are below 1.5 for more than 10 time steps the load has stopped
                end_check.append(i)
            if len(end_check) > 10:
                active = False
                list_of_load_inds.append(new_load[:-10])
                new_load = []
                end_check = []
        elif active == False:    
            if time_series[i] > 10:
                active = True
                if i < 10:
                    start_index = 0
                else:
                    start_index = i-10
                new_load.append(start_index)
    for load in list_of_load_inds:
        list_of_loads.append(time_series[load])
    return list_of_loads

def padding(list_of_loads, length):
    list_of_padded_loads = []
    for load in list_of_loads:
        if len(load) >= length:
            list_of_padded_loads.append(np.array(load[:length]))
        elif len(load) < length:
            list_of_padded_loads.append(np.append(np.array(load), np.zeros(length-len(load))))
    return np.array(list_of_padded_loads)

def find_anomalous_lengths(list_of_loads):
    model=IsolationForest(contamination=0.01)
    model.fit([[len(load)] for load in list_of_loads])
    predictions = model.predict([[len(load)] for load in list_of_loads])
    anomalous_length_indices = [i for i in range(len(list_of_loads)) if predictions[i]==-1]
    return anomalous_length_indices

def train_test(data_list, loads, anomalies):
    data_series = pd.Series(data=[data_point for _, data_point in data_list], index=[timestamp for timestamp, _ in data_list]).sort_index()
    data_series = data_series[~data_series.index.duplicated(keep='first')]
    if loads==[]:
        old_number_of_loads=0
        loads = extract_loads(data_series)
    else:
        old_number_of_loads = len(loads)
        last_load = loads[-1]
        endpoint_last_load = last_load.index[-1]
        loads += extract_loads(data_series.loc[endpoint_last_load:])
    if len(loads) > old_number_of_loads:
        list_of_normalized_loads = [preprocessing.normalize_data(load) for load in loads]
        anomalous_length_indices = find_anomalous_lengths(list_of_normalized_loads)
        if len(loads)-1 in anomalous_length_indices:
            anomalies.append((loads[-1],'length of load'))
            print('A load of anomalous length just ended!')
            return 'load_device_anomaly_length', loads, anomalies
        array_of_normalized_loads = padding(list_of_normalized_loads, max([len(load) for load in list_of_normalized_loads]))
        model=IsolationForest(contamination=0.01)
        model.fit(array_of_normalized_loads)
        predictions = model.predict(array_of_normalized_loads)
        if predictions[-1] < 0:
            anomalies.append((loads[-1],'load'))
            print('A load with an anomalous power curve just ended!')
            return 'load_device_anomaly_power_curve', loads, anomalies
        return None, loads, anomalies
    return None, loads, anomalies
    

    