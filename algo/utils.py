import pandas as pd
import numpy as np
import pickle
import torch
import os
from pyarrow.lib import ArrowInvalid

import operator_lib.util as util

__all__ = ("todatetime", "save_data", "calculate_std", "calculate_mean")

FILE_NAME_OPERATOR_START_TIME = "operator_start_time.pickle"
FILE_NAME_INIT_PHASE_RESET = "init_phase_was_resetted.pickle"

class StdPointOutlierDetector():
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        self.filename_dict = {"current_stddev": f'{data_path}/current_stddev_point.pickle', "current_mean": f'{data_path}/current_mean_point.pickle', 
                              "num_datepoints": f'{data_path}/num_datepoints_point.pickle'}
        
        self.current_stddev = 0
        self.current_mean = 0
        self.num_datepoints = 0

        (self.current_stddev, 
        self.current_mean, 
        self.num_datepoints) = self.load_data(self.current_stddev, 
                                              self.current_mean, 
                                              self.num_datepoints)

        
    def calculate_std(self, new_value, current_stddev, current_mean, num_datepoints):
        current_stddev = np.sqrt(num_datepoints/(num_datepoints + 1)*current_stddev**2 + num_datepoints/((num_datepoints + 1)**2)*(new_value - current_mean)**2)
        return current_stddev
        
    def calculate_mean(self, new_value, current_mean, num_datepoints):
        current_mean = (num_datepoints*current_mean + new_value)/(num_datepoints + 1)
        return current_mean

    def save(self):
        current_stddev_path = self.filename_dict["current_stddev"]
        current_mean_path = self.filename_dict["current_mean"]
        num_datepoints_path = self.filename_dict["num_datepoints"]

        with open(current_stddev_path, 'wb') as f:
            pickle.dump(self.current_stddev, f)
        with open(current_mean_path, 'wb') as f:
            pickle.dump(self.current_mean, f)
        with open(num_datepoints_path, 'wb') as f:
            pickle.dump(self.num_datepoints, f)

    def stop(self):
        util.logger.info("Stop Std Outlier Detector")
        self.save()

    def load_data(self, current_stddev, current_mean, num_datepoints):
        current_stddev_path = self.filename_dict["current_stddev"]
        current_mean_path = self.filename_dict["current_mean"]
        num_datepoints_path = self.filename_dict["num_datepoints"]
        
        if os.path.exists(current_stddev_path):
            with open(current_stddev_path, 'rb') as f:
                current_stddev = pickle.load(f)
        if os.path.exists(current_mean_path):
            with open(current_mean_path, 'rb') as f:
                current_mean = pickle.load(f)
        if os.path.exists(num_datepoints_path):
            with open(num_datepoints_path, 'rb') as f:
                num_datepoints = pickle.load(f)
 
        return current_stddev, current_mean, num_datepoints

    def point_is_anomalous_high(self, point):
        if self.num_datepoints < 2:
            return False
        return point > self.get_upper_threshold()

    def point_is_anomalous_low(self, point):
        if self.num_datepoints < 2:
            return False
        return point < self.get_lower_threshold()

    def get_upper_threshold(self):
        return self.current_mean + 5*self.current_stddev

    def get_lower_threshold(self):
        return self.current_mean - 5*self.current_stddev

    def update(self, point):
        self.current_stddev = self.calculate_std(point, self.current_stddev, self.current_mean, self.num_datepoints)
        self.current_mean = self.calculate_mean(point, self.current_mean, self.num_datepoints)
        self.num_datepoints += 1

def todatetime(timestamp):
        if str(timestamp).isdigit():
            if len(str(timestamp))==13:
                return pd.to_datetime(int(timestamp), unit='ms')
            elif len(str(timestamp))==19:
                return pd.to_datetime(int(timestamp), unit='ns')
        else:
            return pd.to_datetime(timestamp)

def save_data(filename_dict, last_training_time, data_list, model,
              training_performance, anomalies, loads, training_max, reconstruction_errors):
        data_path = filename_dict["data"]
        last_training_time_path = filename_dict["last_training_time"]
        anomalies_path = filename_dict["anomalies"]
        training_performance_path = filename_dict["training_performance"]
        loads_path = filename_dict["loads"]
        model_path = filename_dict["model"]
        training_max_path = filename_dict["training_max"]
        reconstruction_errors_path = filename_dict["reconstruction_errors"]


        data_series = pd.Series(data=[data_point for _, data_point in data_list], index=[timestamp.replace(microsecond=0).strftime('%Y-%m-%d %X') for timestamp, _ in data_list]).sort_index()
        data_series = data_series[~data_series.index.duplicated(keep='first')]
        df = data_series.to_frame()
        df.columns = ['power_values']
        df.to_parquet(data_path)
        with open(last_training_time_path, 'wb') as f:
            pickle.dump(last_training_time, f)
        with open(anomalies_path, 'wb') as f:
            pickle.dump(anomalies, f)
        with open(training_performance_path, 'wb') as f:
            pickle.dump(training_performance, f)
        with open(loads_path, 'wb') as f:
            pickle.dump(loads, f)
        torch.save(model, model_path)
        with open(training_max_path, 'wb') as f:
            pickle.dump(training_max, f)
        with open(reconstruction_errors_path, 'wb') as f:
            pickle.dump(reconstruction_errors, f)


def load_data(filename_dict, data_list, last_training_time, anomalies, training_performance, loads, model, training_max, reconstruction_errors):
    data_path = filename_dict["data"]
    last_training_time_path = filename_dict["last_training_time"]
    anomalies_path = filename_dict["anomalies"]
    training_performance_path = filename_dict["training_performance"]
    loads_path = filename_dict["loads"]
    model_path = filename_dict["model"]
    training_max_path = filename_dict["training_max"]
    reconstruction_errors_path = filename_dict["reconstruction_errors"]

    if os.path.exists(data_path):
        data_list = []
        try:
            df = pd.read_parquet(data_path)
            df.index = pd.to_datetime(df.index)
            data_series = pd.Series(data=df['power_values'], index=df.index)
            data_series = df[~df.index.duplicated(keep='first')]
            for i in range(len(data_series.index)):
                data_list.append([data_series.index[i], float(data_series.iloc[i])])
        except ArrowInvalid:
            print("Data buffer could not be loaded! This might be caused by not having any data in the buffer yet.")

    if os.path.exists(last_training_time_path):
       with open(last_training_time_path, 'rb') as f:
           last_training_time = pickle.load(f)

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

    if os.path.exists(training_max_path):
       with open(training_max_path, 'rb') as f:
           training_max = pickle.load(f)

    if os.path.exists(reconstruction_errors_path):
       with open(reconstruction_errors_path, 'rb') as f:
           reconstruction_errors = pickle.load(f)

    return data_list, last_training_time, anomalies, training_performance, loads, model, training_max, reconstruction_errors

def load(data_path, file_name):
    file_path = os.path.join(data_path, file_name)
    if not os.path.exists(file_path):
        return None 
    with open(file_path, 'rb') as f:
        timestamp = pickle.load(f)
        return timestamp

def save(data_path, file_name, value):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    file_path = os.path.join(data_path, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(value, f)

def load_operator_start_time(data_path):
    return load(data_path, FILE_NAME_OPERATOR_START_TIME)

def save_operator_start_time(data_path, timestamp):
    save(data_path, FILE_NAME_OPERATOR_START_TIME, timestamp)

def load_init_phase_was_resetted(data_path):
    return load(data_path, FILE_NAME_INIT_PHASE_RESET)

def save_init_phase_was_resetted(data_path, init_phase_was_resetted):
    save(data_path, FILE_NAME_INIT_PHASE_RESET, init_phase_was_resetted)

