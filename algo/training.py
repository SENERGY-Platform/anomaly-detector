"""
   Copyright 2026 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

__all__ = ("fit_anomaly_model", "train_anomaly_model")

import datetime
import typing

import pandas as pd
import ray

from . import preprocessing
from .detectors import continuous_curve, extreme_value, load_curve, receive_time
from .history import _read_history
from .model import SCHEMA_VERSION, AnomalyModel, DeviceProfile


def _profile(series: pd.Series, train_curves: bool, epochs: int, log: typing.Callable[[str], None]) -> DeviceProfile:
    baseline = float(series.median())
    device_type = "load_device" if preprocessing.is_load_device(series) else "cont_device"
    values = extreme_value._value_stats(series.to_numpy(float))
    curve = None
    if train_curves:
        curve = (continuous_curve._cont_curve(series, epochs, log) if device_type == "cont_device"
             else load_curve._load_curve(series, baseline))
    return DeviceProfile(device_type, baseline, values, receive_time._gap_stats(series.index), curve)


def fit_anomaly_model(datasets, topic_meta, now: datetime.datetime, train_curves: bool,
                      epochs: int = continuous_curve.EPOCHS) -> typing.Tuple[AnomalyModel, typing.List[str], bool]:
    """
    The model from the input history (None if none was requested), with the log lines and whether a requested
    history could not be read; runs as a Ray task.
    """
    notes: typing.List[str] = []
    history, history_failed = _read_history(datasets, topic_meta, notes)
    devices = {}
    for key, series in history.items():
        try:
            profile = _profile(series, train_curves, epochs, lambda message, key=key: notes.append(f"{key}: {message}"))
        except Exception as ex:
            notes.append(f"{key}: no profile, {type(ex).__name__}: {ex}")
            continue
        devices[key] = profile
        parts = [name for name, part in (("values", profile.values), ("gaps", profile.gaps), ("curve", profile.curve))
                 if part is not None]
        notes.append(f"{key}: {profile.device_type}, {len(series)} values, profile: {', '.join(parts) or 'empty'}")
    return AnomalyModel(SCHEMA_VERSION, now, devices), notes, history_failed


train_anomaly_model = ray.remote(num_cpus=continuous_curve.TORCH_THREADS)(fit_anomaly_model)
