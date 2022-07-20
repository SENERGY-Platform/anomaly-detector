import pandas as pd
import typing

from anom_detector import Anomaly_Detector
import cont_device, load_device


def get_device_type(data_series):
    device_type = 'cont_device'
    for timestamp_1 in data_series.index:
        constantly_zero = True
        for timestamp_2 in data_series.loc[timestamp_1:timestamp_1+pd.Timedelta(2,'hours')]:
            if data_series.loc[timestamp_2] != 0:
                constantly_zero = False
                break
        if constantly_zero == True:
            device_type = 'load_device'
            break    
    return device_type

def init(device_id):
    return Anomaly_Detector(device_id)

def run(anomaly_detector: typing.anom_detector.Anomaly_Detector, data_point, model_file_path='device_id'):
    if anomaly_detector.hist_data_available:
        anomaly_detector.data_series.append(data_point)
        if anomaly_detector.device_type==None:
            if data_point.index-anomaly_detector.initial < pd.Timedelta(1, 'days'):
                pass
            elif data_point.index-anomaly_detector.initial_time >= pd.Timedelta(1, 'days'):
                anomaly_detector.device_type = get_device_type(anomaly_detector)
        if data_point.index-anomaly_detector.last_training_time >= pd.Timedelta(14, 'days'): 
            if anomaly_detector.device_type == 'cont_device':
                if anomaly_detector.last_training_time == anomaly_detector.initial_time:
                    anomaly_detector.model = cont_device.Autoencoder(32)
                anomaly_detector.model = cont_device.batch_training(anomaly_detector.model, anomaly_detector.data_series.loc[data_point.index-pd.Timedelta(14, 'days'):data_point.index], model_file_path)
            elif anomaly_detector.device_type == 'load_device':
                pass # training IsolationForest is that fast, that we can train it again with every new data point.
            anomaly_detector.last_training_time = data_point.index
        elif data_point.index-anomaly_detector.last_training_time < pd.Timedelta(14, 'days'):
            pass
        if anomaly_detector.last_training_time > anomaly_detector.initial_time:
            if anomaly_detector.device_type == 'cont_device':
                cont_device.run(anomaly_detector.data_series, anomaly_detector.model)
            elif anomaly_detector.device_type == 'load_device':
                anomaly_detector.model = load_device.run(anomaly_detector.data_series, model_file_path)
    elif not anomaly_detector.hist_data_available:
        anomaly_detector.data_series.append(data_point)
        anomaly_detector.hist_data_available = True