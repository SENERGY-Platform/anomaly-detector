import os 

from algo.curve_anomaly.cont_det.offline_detector import OfflineTrainContCurveDetector
from algo.curve_anomaly.cont_det.online_detector import OnlineTrainContCurveDetector
from algo.curve_anomaly.load_det.load_detector import LoadCurveDetector

def create_curve_detector(
    mlflow_url, 
    ml_trainer_url, 
    data_path,
    device_type, 
    init_median, 
    first_data_time,
    curve_detector_training_mode,
    device_id,
    operator_id,
    pipeline_id,
    train_level,
    train_interval,
    retrain
):

    data_path = os.path.join(data_path, "curve_explorer")
    if device_type == "cont_device":
        if curve_detector_training_mode == "online":
            return OnlineTrainContCurveDetector(
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
            )
        else:
            return OfflineTrainContCurveDetector(data_path, init_median, first_data_time)
    elif device_type == "load_device":
        return LoadCurveDetector(data_path, init_median, first_data_time)