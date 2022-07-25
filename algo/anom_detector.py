from datetime import datetime

class Anomaly_Detector:
    def __init__(self, device_id):
        self.data = []
        self.initial_time = datetime.utcnow()
        self.first_data_time = None
        self.last_training_time = self.initial_time
        self.device_id = device_id
        self.device_type = None
        self.model = None
        self.anomalies = []

        





