import pandas as pd
from algo import utils
from . import cont_device, load_device
from pyarrow.lib import ArrowInvalid
import pickle
import os

__all__ = ("batch_train", "test")


def batch_train(data_list, first_data_time, last_training_time, device_type, model, use_cuda, training_performance, training_max):
        current_timestamp = utils.todatetime(data_list[-1][0]).tz_localize(None)
        if current_timestamp-last_training_time.tz_localize(None) >= pd.Timedelta(14, 'days'): 
            if device_type == 'cont_device':
                if last_training_time.tz_localize(None) == first_data_time.tz_localize(None):
                    model = cont_device.Autoencoder(32)
                    if use_cuda:
                        model = model.cuda()
                model, training_performance, training_max = cont_device.batch_train(data_list, model, use_cuda, training_performance)
            elif device_type == 'load_device':
                return last_training_time, model, training_performance, training_max
            last_training_time = current_timestamp
            return last_training_time, model, training_performance, training_max
        elif current_timestamp-last_training_time.tz_localize(None) < pd.Timedelta(14, 'days'):
            return last_training_time, model, training_performance, training_max

def test(data_list, first_data_time, last_training_time, device_type, model, use_cuda, anomalies, loads, init_median, reconstruction_errors, training_max):
        if device_type == 'cont_device' and last_training_time.tz_localize(None) > first_data_time.tz_localize(None):
            output, anomalies, reconstruction_errors = cont_device.test(data_list, model, use_cuda, anomalies, training_max, reconstruction_errors)
            return output, loads, anomalies, reconstruction_errors
        elif device_type == 'load_device':
            output, loads, anomalies = load_device.train_test(data_list, loads, anomalies, init_median)
            return output,  loads, anomalies, reconstruction_errors
        else:
            return None, loads, anomalies, reconstruction_errors
        
def save_data(filename_dict, data_list, anomalies):
        data_path = filename_dict["data"]
        anomalies_path = filename_dict["anomalies"]

        data_series = pd.Series(data=[data_point for _, data_point in data_list], index=[timestamp.replace(microsecond=0).strftime('%Y-%m-%d %X') for timestamp, _ in data_list]).sort_index()
        data_series = data_series[~data_series.index.duplicated(keep='first')]
        df = data_series.to_frame()
        df.columns = ['power_values']
        df.to_parquet(data_path)

        with open(anomalies_path, 'wb') as f:
            pickle.dump(anomalies, f)

def load_data(filename_dict, data_list, anomalies):
    data_path = filename_dict["data"]
    anomalies_path = filename_dict["anomalies"]

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

    if os.path.exists(anomalies_path):
       with open(anomalies_path, 'rb') as f:
           anomalies = pickle.load(f)

    return data_list, anomalies