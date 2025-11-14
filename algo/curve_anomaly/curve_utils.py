import pandas as pd
from pyarrow.lib import ArrowInvalid
import pickle
import os

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
        try:
            with open(anomalies_path, 'rb') as f:
                anomalies = pickle.load(f)
        except EOFError:
            print("Could not load the historic anomalies!")
        

    return data_list, anomalies