import pandas as pd
import pickle
import torch
import os

__all__ = ("todatetime", "save_data")

def todatetime(timestamp):
        if str(timestamp).isdigit():
            if len(str(timestamp))==13:
                return pd.to_datetime(int(timestamp), unit='ms')
            elif len(str(timestamp))==19:
                return pd.to_datetime(int(timestamp), unit='ns')
        else:
            return pd.to_datetime(timestamp)

def save_data(filename_dict, initial_time, first_data_time, last_training_time, data_list, model,
              training_performance, anomalies, device_type, loads):
        data_path = filename_dict["data"]
        initial_time_path = filename_dict["initial_time"]
        first_data_time_path = filename_dict["first_data_time"]
        last_training_time_path = filename_dict["last_training_time"]
        device_type_path = filename_dict["device_type"]
        anomalies_path = filename_dict["anomalies"]
        training_performance_path = filename_dict["training_performance"]
        loads_path = filename_dict["loads"]
        model_path = filename_dict["model"]


        data_series = pd.Series(data=[data_point for _, data_point in data_list], index=[timestamp.replace(microsecond=0).strftime('%Y-%m-%d %X') for timestamp, _ in data_list]).sort_index()
        data_series = data_series[~data_series.index.duplicated(keep='first')]
        df = data_series.to_frame()
        df.columns = ['power_values']
        df.to_parquet(data_path)
        with open(initial_time_path, 'wb') as f:
            pickle.dump(initial_time, f)
        with open(first_data_time_path, 'wb') as f:
            pickle.dump(first_data_time, f)
        with open(last_training_time_path, 'wb') as f:
            pickle.dump(last_training_time, f)
        with open(device_type_path, 'wb') as f:
            pickle.dump(device_type, f)
        with open(anomalies_path, 'wb') as f:
            pickle.dump(anomalies, f)
        with open(training_performance_path, 'wb') as f:
            pickle.dump(training_performance, f)
        with open(loads_path, 'wb') as f:
            pickle.dump(loads, f)
        torch.save(model, model_path)


def load_data(filename_dict, data_list, initial_time, first_data_time, last_training_time, device_type, anomalies, training_performance, loads, model):
    data_path = filename_dict["data"]
    initial_time_path = filename_dict["initial_time"]
    first_data_time_path = filename_dict["first_data_time"]
    last_training_time_path = filename_dict["last_training_time"]
    device_type_path = filename_dict["device_type"]
    anomalies_path = filename_dict["anomalies"]
    training_performance_path = filename_dict["training_performance"]
    loads_path = filename_dict["loads"]
    model_path = filename_dict["model"]

    if os.path.exists(data_path):
        data_list = []
        df = pd.read_parquet(data_path)
        df.index = pd.to_datetime(df.index)
        data_series = pd.Series(data=df['power_values'], index=df.index)
        data_series = df[~df.index.duplicated(keep='first')]
        for i in range(len(data_series.index)):
            data_list.append([data_series.index[i], float(data_series.iloc[i])])

    if os.path.exists(initial_time_path):
       with open(initial_time_path, 'rb') as f:
           initial_time = pickle.load(f)

    if os.path.exists(first_data_time_path):
       with open(first_data_time_path, 'rb') as f:
           first_data_time = pickle.load(f)

    if os.path.exists(last_training_time_path):
       with open(last_training_time_path, 'rb') as f:
           last_training_time = pickle.load(f)

    if os.path.exists(device_type_path):
       with open(device_type_path, 'rb') as f:
           device_type = pickle.load(f)

    if os.path.exists(anomalies_path):
       with open(anomalies_path, 'rb') as f:
           anomalies = pickle.load(f)

    if os.path.exists(training_performance_path):
       with open(training_performance_path, 'rb') as f:
           training_performance = pickle.load(f)

    if os.path.exists(loads_path):
       with open(loads_path, 'rb') as f:
           loads = pickle.load(f)

    if os.path.exists(model_path):
        model = torch.load(model_path)

    return data_list, initial_time, first_data_time, last_training_time, device_type, anomalies, training_performance, loads, model