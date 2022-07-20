import pandas as pd
from datetime import datetime

class Anomaly_Detector:
    def __init__(self, device_id):
        self.data_series = pd.Series([])
        self.initial_time = datetime.min
        self.device_id = device_id
        self.device_type = None
        self.model = None

        self.last_training_time = self.initial_time

        if database[self.device_id]:
            self.hist_data_available = True
            self.data_series = database[self.device_id]
            self.initial_time = self.data_series.index.min
        else:
            self.hist_data_available = False





