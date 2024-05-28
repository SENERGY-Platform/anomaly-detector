from algo.curve_anomaly.cont_device.cont_detector import ContCurveDetector

class OnlineTrainContCurveDetector(ContCurveDetector):
    # Used for outsourced training of the model 
    def __init__(self, data_path, device_type, init_median, first_data_time):
        super().__init__(data_path, device_type, init_median, first_data_time)