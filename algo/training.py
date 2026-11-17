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

__all__ = ("fit_anomaly_model", "train_anomaly_model", "profile_table")

import datetime
import typing

import numpy as np
import pandas as pd
import ray

from .history import _read_history
from .model import SCHEMA_VERSION, AnomalyModel, DeviceProfile
from .stats import gap_stats, value_stats


def _profile(series: pd.Series) -> DeviceProfile:
    gaps_min = np.diff(series.index.to_numpy()) / np.timedelta64(1, "m")
    return DeviceProfile(value_stats(series.to_numpy(float)), gap_stats(gaps_min))


def fit_anomaly_model(datasets, topic_meta, now: datetime.datetime) -> typing.Tuple[AnomalyModel, typing.List[str], bool]:
    """
    The model from the input history (None if none was requested), with the log lines and whether a requested
    history could not be read; runs as a Ray task.
    """
    notes: typing.List[str] = []
    history, history_failed = _read_history(datasets, topic_meta, notes)
    devices = {}
    for key, series in history.items():
        try:
            profile = _profile(series)
        except Exception as ex:
            notes.append(f"{key}: no profile, {type(ex).__name__}: {ex}")
            continue
        devices[key] = profile
        parts = [name for name, part in (("values", profile.values), ("gaps", profile.gaps)) if part is not None]
        notes.append(f"{key}: {len(series)} values, profile: {', '.join(parts) or 'empty'}")
    return AnomalyModel(SCHEMA_VERSION, now, devices), notes, history_failed



train_anomaly_model = ray.remote(fit_anomaly_model)


def profile_table(model: AnomalyModel) -> pd.DataFrame:
    """The bounds of every device, one row each, as the training logs them to MLflow."""
    rows = []
    for key, p in sorted(model.devices.items()):
        rows.append({
            "device_id": key,
            "value_mean": p.values.mean if p.values else None,
            "value_lower": p.values.lower() if p.values else None,
            "value_upper": p.values.upper() if p.values else None,
            "gap_mean_min": p.gaps.mean_min if p.gaps else None,
            "gap_upper_min": p.gaps.upper() if p.gaps else None,
        })
    return pd.DataFrame(rows, columns=["device_id", "value_mean", "value_lower", "value_upper", "gap_mean_min",
                                       "gap_upper_min"])
