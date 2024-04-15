import os 

import operator_lib.util as util

from algo import curve_anomaly
from algo import point_outlier
from algo import consumption_anomaly
from algo import utils
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
        produce_func
    ):
        self.active_detectors = []
        
        if check_data_schema:
            util.logger.info(f"{LOG_PREFIX}: Data Schema Detector is active")

        if check_data_anomalies:
            print(f"{LOG_PREFIX}: Curve Explorer is active!")
            self.Curve_Explorer = curve_anomaly.Curve_Explorer(data_path)
            self.active_detectors.append(self.Curve_Explorer)
        
        if check_data_extreme_outlier:
            util.logger.info(f"{LOG_PREFIX}: Point Explorer is active!")
            self.Point_Explorer = point_outlier.Point_Explorer(os.path.join(data_path, "point_explorer"))
            self.active_detectors.append(self.Point_Explorer)

        self.frequency_monitor = None
        if check_receive_time_outlier:
            util.logger.info(f"{LOG_PREFIX}: Frequency Monitor is active!")
            self.frequency_monitor = FrequencyDetector(
                kafka_produce_func=produce_func,
                data_path=os.path.join(data_path, "frequency_monitor")
            )
            self.frequency_monitor.start()

        if check_consumption:
            consumption_explorer = consumption_anomaly.Consumption_Explorer(os.path.join(data_path, "consumption_explorer"))


    def check_input(self, value, timestamp):
        anomaly_results = []
        for detector in self.active_detectors:
            sample_is_anomalous, result = detector.check(value, timestamp)

            if sample_is_anomalous:
                result['device_id'] = self.device_id
                result['initial_phase'] = ''
                util.logger.info(f"{LOG_PREFIX}: Anomaly occured: {result}")
                anomaly_results.append(result) 
        return anomaly_results

    def update(self, value, timestamp, real_time):
        # Update detectors
        for detector in self.active_detectors:
            detector.update_with_new_value(value)

        if self.frequency_monitor and real_time:
            # Historic data comes not with pauses in between
            self.frequency_monitor.register_input(timestamp)

    def start_freq_loop(self):
        if not self.frequency_monitor:
            return 

        self.frequency_monitor.start_loop()

    def stop(self):
        if self.frequency_monitor:
            self.frequency_monitor.stop()
            self.frequency_monitor.save()

        for active_detector in self.active_detectors:
            active_detector.stop()
