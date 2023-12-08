import pandas as pd
from algo import utils
from . import cont_device, load_device

__all__ = ("get_device_type", "batch_train", "test")

def get_device_type(data_list):# entries in data_list are of the form (timestamp, data point)
        data_series = pd.Series(data=[data_point for _, data_point in data_list], index=[timestamp for timestamp, _ in data_list]).sort_index()
        data_series = data_series[~data_series.index.duplicated(keep='first')]
        device_type = 'cont_device'
        for timestamp_1 in data_series.index:
            constantly_zero = True
            if timestamp_1 + pd.Timedelta(2,'hours') < data_series.index.max():
                for timestamp_2 in data_series.loc[timestamp_1:timestamp_1+pd.Timedelta(2,'hours')].index:
                    if data_series.loc[timestamp_2] > 5:
                        constantly_zero = False
                        break
                if constantly_zero == True:
                    device_type = 'load_device'
                    break    
        return device_type

def batch_train(data, first_data_time, last_training_time, device_type, model, use_cuda, training_performance):
        if utils.todatetime(data['energy_time']).tz_localize(None)-last_training_time >= pd.Timedelta(14, 'days'): 
            if device_type == 'cont_device':
                if last_training_time == first_data_time:
                    model = cont_device.Autoencoder(32)
                    if use_cuda:
                        model = model.cuda()
                model, training_performance = cont_device.batch_train(data, model, use_cuda, training_performance)
            elif device_type == 'load_device':
                pass # training IsolationForest is that fast, that we can train it again with every new data point.
            last_training_time = utils.todatetime(data['energy_time']).tz_localize(None)
            return last_training_time, model, training_performance
        elif utils.todatetime(data['energy_time']).tz_localize(None)-last_training_time < pd.Timedelta(14, 'days'):
            pass

def test(data, first_data_time, last_training_time, device_type, model, use_cuda, anomalies, loads):
        if device_type == 'cont_device' and last_training_time > first_data_time:
            output, anomalies = cont_device.test(data, model, use_cuda, anomalies)
            return output, loads, anomalies
        elif device_type == 'load_device':
            output, loads, anomalies = load_device.train_test(data, loads, anomalies)
            return output,  loads, anomalies
        else:
            return None, loads, anomalies