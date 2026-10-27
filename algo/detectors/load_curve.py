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

Detects a run of a load device whose length or curve deviates from the runs in its history. A run is the time the device works above its rest level, and the runs are only judged for a device whose runs are regular.
"""

import dataclasses
import datetime
import typing

import numpy as np
import pandas as pd

from .. import preprocessing
from ..model import K_SIGMA

# Floors under the robust bounds of a load device: with regular runs the MAD is close to zero, and a bound of
# a few MADs would flag every run that differs by a minute or by sensor noise.
LENGTH_MIN_TOLERANCE = 5  # minutes
CURVE_MIN_TOLERANCE = 0.05  # normalised units
MAD_TO_SIGMA = 1.4826
# Runs are only judged for a device whose runs are mostly of one length: this share of them within this share of
# the median length (at least LENGTH_MIN_TOLERANCE). A fixed width, because a width from the MAD grows with the
# spread and would call any device regular.
LENGTH_REGULAR_SHARE = 0.8
LENGTH_REGULAR_WIDTH = 0.25
MIN_LOADS = 15
# A device that never leaves its run must not grow the buffer without bound; the length check
# already fires for a run this long.
MAX_RUN_SAMPLES = 100_000
RUN_START_ABOVE = 10
RUN_END_AT_MOST = 1
RUN_END_SPAN = datetime.timedelta(minutes=10)


def pad_load(load: np.ndarray, length: int) -> np.ndarray:
    """A load cut or zero-padded to length."""
    load = np.asarray(load, dtype=float).reshape(-1)
    if len(load) >= length:
        return load[:length]
    return np.append(load, np.zeros(length - len(load)))


@dataclasses.dataclass
class LoadCurveModel:
    """
    Robust deviation instead of a learned outlier model: a bound from the median and MAD of the training runs
    also holds for values outside the range seen in training.
    """
    length_median: float
    length_mad: float
    median_curve: np.ndarray  # elementwise median of the normalised training runs, padded to pad_length
    dist_median: float  # median of the mean absolute distance of the training runs to median_curve
    dist_mad: float
    pad_length: int
    training_max: float
    n_loads: int
    # Whether the training runs are of one length; a device like a TV, with blips of minutes and evenings of
    # hours, has neither a length nor a curve to deviate from, so its runs are not judged.
    regular: bool

    def length_tolerance(self) -> float:
        return max(LENGTH_MIN_TOLERANCE, K_SIGMA * MAD_TO_SIGMA * self.length_mad)

    def curve_distance(self, load: np.ndarray) -> float:
        """Mean absolute distance of a run in the original unit to the median curve."""
        curve = pad_load(np.asarray(load, dtype=float).reshape(-1) / self.training_max, self.pad_length)
        return float(np.abs(curve - self.median_curve).mean())

    def score(self, load: np.ndarray) -> typing.Optional[str]:
        """load: the minute values of a finished run in the original unit."""
        load = np.asarray(load, dtype=float).reshape(-1)
        if not self.regular:
            return None
        if abs(len(load) - self.length_median) > self.length_tolerance():
            return "uncontinious_device_length"
        if self.curve_distance(load) - self.dist_median > max(CURVE_MIN_TOLERANCE,
                                                              K_SIGMA * MAD_TO_SIGMA * self.dist_mad):
            return "uncontinious_device_curve"
        return None


class RunTracker:
    """
    Follows the runs of a load device with the rule of preprocessing.active_mask: a run starts above
    baseline+10 (taking the sample before it along) and ends with the first sample at or below
    baseline+1 that follows ten minutes of samples all at or below it.
    """

    def __init__(self):
        self.active = False
        self.below: typing.Optional[datetime.datetime] = None
        self.prev: typing.Optional[preprocessing.Sample] = None
        self.samples: typing.List[preprocessing.Sample] = []

    def update(self, ts: datetime.datetime, value: float, baseline: float
               ) -> typing.Tuple[bool, typing.Optional[typing.List[preprocessing.Sample]]]:
        """Whether the sample lies in a run, and the samples of the run it finishes."""
        finished = None
        in_run = False
        if not self.active:
            if value > baseline + RUN_START_ABOVE:
                self.active = True
                self.below = None
                self.samples = [self.prev] if self.prev is not None else []
                self.samples.append((ts, value))
                in_run = True
        else:
            in_run = True
            if len(self.samples) < MAX_RUN_SAMPLES:
                self.samples.append((ts, value))
            if value <= baseline + RUN_END_AT_MOST:
                if self.below is None:
                    self.below = ts
                elif ts - self.below >= RUN_END_SPAN:
                    finished = self.samples
                    self.active = False
                    self.below = None
                    self.samples = []
            else:
                self.below = None
        self.prev = (ts, value)
        return in_run, finished


def fit_load_curve(loads: typing.List[np.ndarray]) -> typing.Optional[LoadCurveModel]:
    """Median and MAD of the length and of the distance to the median curve over the runs."""
    if len(loads) < MIN_LOADS:
        return None
    training_max = max(float(load.max()) for load in loads)
    if not np.isfinite(training_max) or training_max <= 0:
        return None
    pad_length = max(len(load) for load in loads)
    lengths = np.array([len(load) for load in loads], dtype=float)
    curves = np.array([pad_load(load / training_max, pad_length) for load in loads])
    median_curve = np.median(curves, axis=0)
    distances = np.abs(curves - median_curve).mean(axis=1)
    length_median = float(np.median(lengths))
    dist_median = float(np.median(distances))
    length_mad = float(np.median(np.abs(lengths - length_median)))
    model = LoadCurveModel(length_median, length_mad, median_curve, dist_median,
                           float(np.median(np.abs(distances - dist_median))), pad_length, training_max, len(loads),
                           True)
    width = max(LENGTH_MIN_TOLERANCE, LENGTH_REGULAR_WIDTH * length_median)
    model.regular = bool(np.mean(np.abs(lengths - length_median) <= width) >= LENGTH_REGULAR_SHARE)
    return model


def _load_curve(series: pd.Series, baseline: float) -> typing.Optional[LoadCurveModel]:
    loads = [load.to_numpy(float) for load in preprocessing.extract_loads(preprocessing.to_minutes(series), baseline)]
    return fit_load_curve(loads)


def check(curve: LoadCurveModel, key, samples, baseline) -> typing.Optional[dict]:
    minutes = preprocessing.to_minutes(preprocessing.to_series(samples))
    if minutes.empty:
        return None
    # Cut as the training cuts it: the same rule on the same interpolated minutes. A sample-wise cut
    # (the last ten minutes of the samples) is up to a cadence too long for slow devices.
    loads = preprocessing.extract_loads(minutes, baseline)
    # A run capped at MAX_RUN_SAMPLES has no end in the series; the length check fires for it anyway.
    load = loads[0] if loads else minutes[:minutes.index[-1] - pd.Timedelta(10, "min")]
    if load.empty:
        return None
    sub_type = curve.score(load.to_numpy())
    if sub_type is None:
        return None
    return {"type": "curve_anomaly", "sub_type": sub_type, "value": "", "start_time": preprocessing.iso_utc(load.index[0]),
            "end_time": preprocessing.iso_utc(load.index[-1]), "device_id": key}
