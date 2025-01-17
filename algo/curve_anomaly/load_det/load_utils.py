import pickle
import os

def save_data(filename_dict, loads, endpoint_last_load):
        loads_path = filename_dict["loads"]
        endpoint_last_load_path = filename_dict["endpoint_last_load"]

        with open(loads_path, 'wb') as f:
            pickle.dump(loads, f)

        with open(endpoint_last_load_path, 'wb') as f:
            pickle.dump(endpoint_last_load, f)

def load_data(filename_dict):
    loads_path = filename_dict["loads"]
    endpoint_last_load_path = filename_dict["endpoint_last_load"]

    loads = []
    endpoint_last_load = None
    if os.path.exists(loads_path):
       with open(loads_path, 'rb') as f:
           loads = pickle.load(f)

    if os.path.exists(endpoint_last_load_path):
       with open(endpoint_last_load_path, 'rb') as f:
           endpoint_last_load = pickle.load(f)

    return loads, endpoint_last_load

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..cont_det import preprocessing
import operator_lib.util as util

def extract_loads(time_series, init_median):
    list_of_loads = []
    list_of_load_inds = []
    new_load = []
    active = False
    for i in range(len(time_series)):
        if active == True:
            new_load.append(i)
            if time_series[i] < init_median + 1 and not start_of_end: 
                start_of_end = time_series.index[i]
            elif time_series[i] > init_median + 1 and start_of_end:
                start_of_end = None
            elif time_series[i] <= init_median + 1 and start_of_end:
                if time_series.index[i] - start_of_end >= pd.Timedelta(10,"min"): # If values where constantly below the threshold for 10min, the load has stopped.
                    active = False
                    list_of_load_inds.append(new_load)
                    new_load = []
        elif active == False:    
            if time_series[i] > init_median + 10:
                active = True
                if i < 1:
                    new_load = [0]
                else:
                    new_load = [i-1, i]
                start_of_end = None
    for load in list_of_load_inds:
        load_ds = time_series[load]
        list_of_loads.append(load_ds[:load_ds.index[-1]-pd.Timedelta(10,"min")])
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

def train_test(data_list, loads, anomalies, init_median):
    data_series = pd.Series(data=[data_point for _, data_point in data_list], index=[timestamp for timestamp, _ in data_list]).sort_index()
    data_series = data_series[~data_series.index.duplicated(keep='first')]
    if loads==[]:
        old_number_of_loads=0
    else:
        old_number_of_loads = len(loads)
    loads += extract_loads(data_series, init_median)
    if loads==[]:
        endpoint_last_load = data_series.index[0]
    else:
        endpoint_last_load = loads[-1].index[-1]
    if len(loads) > old_number_of_loads:
        list_of_normalized_loads = [preprocessing.normalize_data(load) for load in loads]
        anomalous_length_indices = find_anomalous_lengths(list_of_normalized_loads)
        if len(loads)-1 in anomalous_length_indices and len(loads) >= 15:
            anomalies.append((loads[-1],'length of load'))
            util.logger.debug('A load of anomalous length just ended!')
            return 'load_device_anomaly_length', loads, anomalies, endpoint_last_load
        array_of_normalized_loads = padding(list_of_normalized_loads, max([len(load) for load in list_of_normalized_loads]))
        model=IsolationForest(contamination=0.01)
        model.fit(array_of_normalized_loads)
        predictions = model.predict(array_of_normalized_loads)
        if predictions[-1] < 0:
            anomalies.append((loads[-1],'load'))
            util.logger.debug('A load with an anomalous power curve just ended!')
            return 'load_device_anomaly_power_curve', loads, anomalies, endpoint_last_load
        return None, loads, anomalies, endpoint_last_load
    return None, loads, anomalies, endpoint_last_load
    

    
