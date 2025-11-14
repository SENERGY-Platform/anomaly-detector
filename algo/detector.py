import os 

import operator_lib.util as util
from operator_lib.util import timestamp_to_str

from algo import curve_anomaly
from algo import point_outlier
from algo import consumption_anomaly
from algo.frequency_point_outlier import FrequencyDetector

LOG_PREFIX = "DETECTOR"

class AnomalyDetector():
    def __init__(
        self,
        device_id,
        check_receive_time_outlier,
        check_data_schema,
        check_data_anomalies,
        check_data_extreme_outlier,
        check_consumption,
        data_path,
        produce_func,
        device_type,
        init_median,
        first_data_time,
        ml_trainer_url,
        mlflow_url,
        curve_detector_training_mode,
        operator_id,
        pipeline_id,
        train_level,
        train_interval,
        retrain
    ):
        self.active_detectors = []
        self.device_id = device_id
        self.device_type = device_type
        self.init_median = init_median
        self.first_data_time = first_data_time
        self.check_data_anomalies = check_data_anomalies
        self.mlflow_url = mlflow_url
        self.ml_trainer_url = ml_trainer_url
        self.curve_detector_training_mode = curve_detector_training_mode
        self.data_path = data_path
        self.operator_id = operator_id
        self.pipeline_id = pipeline_id
        self.train_level = train_level
        self.train_interval = train_interval
        self.retrain = retrain 

        if check_data_schema:
            util.logger.info(f"{LOG_PREFIX}: Data Schema Detector is active")
        
        if check_data_extreme_outlier:
            util.logger.info(f"{LOG_PREFIX}: Point Explorer is active!")
            self.Point_Explorer = point_outlier.Point_Explorer(os.path.join(data_path, "point_explorer"), self.device_type, self.init_median)
            self.active_detectors.append(self.Point_Explorer)

        self.frequency_monitor = None
        if check_receive_time_outlier:
            util.logger.info(f"{LOG_PREFIX}: Frequency Monitor is active!")
            self.frequency_monitor = FrequencyDetector(
                kafka_produce_func=produce_func,
                data_path=os.path.join(data_path, "frequency_monitor")
            )
            self.frequency_monitor.start()

        self.frequency_monitor_loop_is_running = False

        if check_consumption:
            consumption_explorer = consumption_anomaly.Consumption_Explorer(os.path.join(data_path, "consumption_explorer"))


    def update_device_type(self, device_type):
        for detector in self.active_detectors:
            detector.device_type = device_type

        if self.check_data_anomalies:
            print(f"{LOG_PREFIX}: Curve Explorer is active!")
            self.Curve_Explorer = curve_anomaly.create_curve_detector(
                self.mlflow_url, 
                self.ml_trainer_url, 
                self.data_path, 
                device_type, 
                self.init_median, 
                self.first_data_time, 
                self.curve_detector_training_mode, 
                self.device_id, 
                self.operator_id, 
                self.pipeline_id,
                self.train_level,
                self.train_interval,
                self.retrain    
            )
            
            self.active_detectors.append(self.Curve_Explorer)

    def update_init_median(self, init_median):
        for detector in self.active_detectors:
            detector.init_median = init_median


    def check_input(self, value, timestamp):
        anomaly_results = []
        for detector in self.active_detectors:
            sample_is_anomalous, result = detector.check(value, timestamp)

            if sample_is_anomalous:
                result['device_id'] = self.device_id
                result['initial_phase'] = ''
                result['timestamp'] = timestamp_to_str(timestamp)
                util.logger.info(f"{LOG_PREFIX}: Anomaly occured: {result}")
                anomaly_results.append(result) 
        return anomaly_results

    def update(self, value, timestamp, real_time):
        # Update detectors
        for detector in self.active_detectors:
            detector.update_with_new_value(value, timestamp)

        if self.frequency_monitor and real_time:
            # Historic data comes not with pauses in between
            self.frequency_monitor.register_input(timestamp)

    def start_freq_loop(self):
        if not self.frequency_monitor:
            return 
        if self.frequency_monitor_loop_is_running:
            return
        self.frequency_monitor.start_loop()

    def stop(self):
        if self.frequency_monitor:
            self.frequency_monitor.stop()
            self.frequency_monitor.save()

        for active_detector in self.active_detectors:
            active_detector.stop()
