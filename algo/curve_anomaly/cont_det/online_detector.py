import operator_lib.util as util
from algo.curve_anomaly.cont_det.cont_detector import ContCurveDetector
import torch
import json
import pandas as pd
from . import cont_device
import requests 
import mlflow 
from operator_lib.util.persistence import save, load

JOB_ID_FILENAME = "training_job_id.pickle"

class OnlineTrainContCurveDetector(ContCurveDetector):
    # Used for outsourced training of the model 
    def __init__(
        self, 
        data_path, 
        init_median, 
        first_data_time, 
        ml_trainer_url, 
        device_id, 
        mlflow_url,
        operator_id,
        pipeline_id,
        train_level,
        train_interval,
        retrain
    ):
        super().__init__(data_path, init_median, first_data_time)
        self.ml_trainer_url = ml_trainer_url
        self.device_id = device_id
        self.mlflow_url = mlflow_url
        self.model = None
        self.job_id = load(data_path, JOB_ID_FILENAME)
        self.operator_id = operator_id
        self.pipeline_id = pipeline_id
        self.anomalies = []
        self.train_level = train_level
        self.train_interval = train_interval
        self.retrain = retrain 
        self.training_is_running = False

    def check(self, value, timestamp):
        if self.first_data_time == None:
            self.first_data_time = timestamp
            self.last_training_time = self.first_data_time
            self.data_list.append([timestamp, value])
            return False, ''
        self.data_list.append([timestamp, value])
        if self.data_list[-1][0] - self.data_list[0][0] >= pd.Timedelta(10, "h"): # Only keep data, which is at most 10 hours old.
            del self.data_list[0]

        if self.training_shall_start(timestamp):
            self.start_training(timestamp)

        if self.model_shall_be_downloaded():
            self.load_model()
        
        if not self.model:
            util.logger.debug("Model is not available. Skip inference")
            return False, ''

        util.logger.debug(f"First Data Point: {self.data_list[0]} - Last Data Point: {self.data_list[-1]}")
        if self.data_list[-1][0] - self.data_list[0][0] < pd.Timedelta(4, "h"):
            util.logger.debug("Not enough data for inference. Need at least 4 hours")
            return False, ''

        anomalies, self.reconstruction_errors = self.test(self.data_list, self.model)
        if not anomalies:
            return False, ''

        self.anomalies.append(anomalies)

        time_window_start = (timestamp-pd.Timedelta(1,'hour')).floor('min')
        self.timestamp_last_anomaly, anomaly_during_last_30_min = cont_device.notification_decision(self.timestamp_last_anomaly, timestamp)
        
        if anomaly_during_last_30_min:
            return True, self.create_result(f'In der Zeit seit {str(time_window_start)} wurde eine Anomalie im Lastprofil festgestellt.', str(time_window_start), "continous_device")
        
        return False, ''
            
    def training_shall_start(self, timestamp):
        # Training shall start when there is enough initial data or when retraining is enabled
        util.logger.debug(f"Current Time: {timestamp} - Last Train Time: {self.last_training_time} < {self.train_interval}{self.train_level}")
        if timestamp - self.last_training_time < pd.Timedelta(self.train_interval, self.train_level):
            util.logger.debug("Wait with training until enough data is collected")
            return False 

        if not self.job_id:
            util.logger.debug("No existing JobID -> Start first training")
            return True 

        if self.retrain:
            util.logger.debug("Retrain Period over. Start new training.")
            return True

        return False

    def start_training(self, timestamp):
        topic_name, path_to_time, path_to_value = self._get_input_topic()
        job_request = {
            "task": "anomaly_detection",
            "task_settings": {
                "model_parameter": {
                    "window_length": 205,
                    "batch_size": 1,
                    "lr": 0.0001,
                    "num_epochs": 20,
                    "loss": "MSE",
                    "op": "Adam",
                    "latent_dims": 32,
                    "early_stopping_patience": 0,
                    "early_stopping_delta": 0,
                    "kernel_size": 7
                },
                "model_name": "cnn"
            },
            "experiment_name": "",
            "data_source": "kafka",
            "data_settings": {
                "name": topic_name,
                "path_to_time": path_to_time,
                "path_to_value": path_to_value,
                "filterType": "device_id",
                "filterValue": self.device_id,
                "ksql_url": "http://ksql.kafka-sql:8088",
                "timestamp_format": "unix", #yyyy-MM-ddTHH:mm:ss.SSSZ
                "time_range_value": "1",
                "time_range_level": "d"
            },
            "toolbox_version": "v2.2.71",
            "ray_image": "ghcr.io/senergy-platform/ray:v0.0.8",
            "ray_version": "2.0.9", # must be the same as in the image
            "cluster": {
                "number_workers": 1,
                "cpu_worker_limit": 2
            }
        }
        util.logger.debug(f"Start online training")
        res = requests.post(self.ml_trainer_url + "/mlfit", json=job_request)
        util.logger.debug(f"ML Trainer Response: {res.text}")
        if res.status_code != 200:
            util.logger.error(f"Cant start training job {res.text}")
            return
        self.job_id = res.json()['task_id']
        util.logger.debug(f"Created Training Job with ID: {self.job_id}")
        self.last_training_time = timestamp
        save(self.data_path, JOB_ID_FILENAME, self.job_id)
        self.training_is_running = True

    def is_job_ready(self):
        res = requests.get(self.ml_trainer_url + "/job/"+self.job_id)
        res_data = res.json()
        job_status = res_data['success'] 
        util.logger.debug(f"Training Job Status: {job_status}")
        if job_status == 'error':
            raise Exception(res_data['response'])

        return job_status == 'done'
    
    def model_shall_be_downloaded(self): 
        if not self.training_is_running:
            util.logger.debug("No need to download model. No training running.")
            return False 

        if self.job_id and self.is_job_ready():
            util.logger.debug("Training Job is ready -> model can be downloaded")
            return True

        util.logger.debug("Training Job is not ready yet")
        return False

    def load_model(self):
        mlflow.set_tracking_uri(self.mlflow_url)
        model_uri = f"models:/{self.job_id}@production"
        util.logger.debug(f"Try to download model {self.job_id}")
        self.model = mlflow.pyfunc.load_model(model_uri)
        util.logger.debug(f"Downloading model {self.job_id} was succesfull")
        unwrapped_model = self.model.unwrap_python_model()
        unwrapped_model.set_all_reconstruction_errors(self.reconstruction_errors)
        self.training_is_running = False
        
    def test(self, data_list, model):
        util.logger.debug("Run inference")
        data_series = pd.DataFrame({"value": [data_point for _, data_point in data_list], "time": [timestamp.replace(microsecond=0) for timestamp, _ in data_list]}).sort_values(by="time")
        reconstruction, reconstruction_error_is_anomalous, anomalous_time_window, anomalous_time_window_smooth, all_reconstruction_errors = model.predict(data_series)
        if reconstruction_error_is_anomalous:
            return ((anomalous_time_window, anomalous_time_window_smooth, reconstruction), all_reconstruction_errors)
        return (None, None)

    def _get_input_topic(self):
        dep_config = util.DeploymentConfig()
        config_json = json.loads(dep_config.config)
        opr_config = util.OperatorConfig(config_json)
        topic_name = None
        path_to_time = None 
        path_to_value = None
        for input_topic in opr_config.inputTopics:
            if self.device_id in input_topic.filterValue.split(','):
                topic_name = input_topic.name
                for mapping in input_topic.mappings:
                    if mapping.dest == "value":
                        path_to_value = mapping.source
                    
                    if mapping.dest == "time":
                        path_to_time = mapping.source

        return topic_name, path_to_time, path_to_value

    def stop(self):
        if self.model:
            self.reconstruction_errors = self.model.unwrap_python_model().get_all_reconstruction_errors()
        super().stop()