import numpy as np
from sklearn.ensemble import IsolationForest
import pickle

import preprocessing

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
            list_of_padded_loads.append(np.append(load, np.zeros(length-len(load))))
    return np.array(list_of_padded_loads)

def find_anomalous_lengths(list_of_loads):
    model=IsolationForest(contamination=0.01)
    model.fit([[len(load)] for load in list_of_loads])
    anomalous_length_indices = model.predict([[len(load)] for load in list_of_loads])
    return anomalous_length_indices

def train_test(data_series, model_file_path):
    list_of_loads = extract_loads(data_series)
    list_of_normalized_loads = [preprocessing.normalize_data(load) for load in list_of_loads]
    anomalous_length_indices = find_anomalous_lengths(list_of_normalized_loads)
    if len(list_of_loads)-1 in anomalous_length_indices:
        return # output 'There just was a load of anomalous length!'
    array_of_loads = padding(list_of_loads, max([len(load) for load in list_of_loads]))
    model=IsolationForest()
    model.fit(array_of_loads[-50:])
    with open(model_file_path, 'wb') as f:
        pickle.dump(model, f)
    predictions = model.predict(array_of_loads[-50:])
    if predictions[-1] < 0:
        return # output 'The last load was anomalous'
    

    