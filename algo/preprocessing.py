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

__all__ = ("to_minutes", "smooth", "windows", "is_load_device", "extract_loads", "active_mask",
           "to_series", "iso_utc")

import datetime
import typing

import numpy as np
import pandas as pd

from .model import WINDOW

# A load device rests (5 % quantile) at most this share of its working level (95 % quantile) ...
IDLE_SHARE = 0.1
# ... and spends at least this share of the time within a tenth of the way from rest to work.
IDLE_TIME_SHARE = 0.5
RUN_END_SPAN = np.timedelta64(10, "m")


def _clean(series: pd.Series) -> pd.Series:
    series = series.astype(float).sort_index(kind="stable")
    return series[~series.index.duplicated(keep="first")]


def to_minutes(series: pd.Series) -> pd.Series:
    """Linear interpolation at the full minutes, without a per-second grid."""
    series = _clean(series)
    if series.empty:
        return series
    minutes = pd.date_range(series.index[0].ceil("min"), series.index[-1].floor("min"), freq="min")
    if len(minutes) == 0:
        return series.iloc[:0]
    dense = series.reindex(series.index.union(minutes)).interpolate(method="time")
    return dense.loc[minutes]


def smooth(series: pd.Series) -> pd.Series:
    # The edges stay NaN: filling them with 0 would look like a drop to zero to the autoencoder.
    return series.rolling(window=30, win_type="exponential", center=True).mean()


def windows(values: np.ndarray, length: int = WINDOW, step: int = 10) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) < length:
        return np.empty((0, length), dtype=np.float32)
    return np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(values, length)[::step])


def is_load_device(series: pd.Series) -> bool:
    """
    Whether the device rests near zero most of the time: the 5 % quantile of the minute values is not negative,
    at most IDLE_SHARE of the 95 % quantile, and the values stay near it at least IDLE_TIME_SHARE of the time.
    A quantity around zero, such as a temperature, is no load device.
    """
    minutes = to_minutes(series)
    if minutes.empty:
        return False
    values = minutes.to_numpy()
    idle, work = np.quantile(values, [0.05, 0.95])
    if work <= idle:
        # A device that is mostly at rest has its 95 % quantile at rest too; its runs show in the maximum.
        work = float(values.max())
    if idle < 0 or idle > IDLE_SHARE * work:
        return False
    return bool(np.mean(values <= idle + 0.1 * (work - idle)) >= IDLE_TIME_SHARE)


def _runs(series: pd.Series, baseline: float) -> typing.Tuple[typing.List[typing.Tuple[int, int]], typing.Optional[int]]:
    """Positions (start, end) of the finished runs and the start of one still running at the end."""
    v = series.to_numpy(dtype=float)
    t = series.index.to_numpy()
    runs = []
    start = None
    below = None
    for i in range(len(v)):
        if start is None:
            if v[i] > baseline + 10:
                start = max(i - 1, 0)
                below = None
        elif v[i] <= baseline + 1:
            if below is None:
                below = t[i]
            elif t[i] - below >= RUN_END_SPAN:
                runs.append((start, i))
                start = None
        else:
            below = None
    return runs, start


def extract_loads(series: pd.Series, baseline: float) -> typing.List[pd.Series]:
    """Finished runs, without the last 10 minutes that only proved the end; a run still going is not returned."""
    series = _clean(series)
    loads = []
    for start, end in _runs(series, baseline)[0]:
        load = series.iloc[start:end + 1]
        loads.append(load[:load.index[-1] - pd.Timedelta(10, "min")])
    return loads


def active_mask(series: pd.Series, baseline: float) -> pd.Series:
    """True inside runs, including one still going at the end."""
    series = _clean(series)
    runs, open_start = _runs(series, baseline)
    mask = np.zeros(len(series), dtype=bool)
    for start, end in runs:
        mask[start:end + 1] = True
    if open_start is not None:
        mask[open_start:] = True
    return pd.Series(mask, index=series.index)


Sample = typing.Tuple[datetime.datetime, float]


def to_series(samples: typing.Iterable[Sample]):
    """The samples as a series on a naive UTC index, the form the preprocessing and the training use."""
    times, values = [], []
    for t, v in samples:
        times.append(t.astimezone(datetime.timezone.utc).replace(tzinfo=None))
        values.append(v)
    return pd.Series(values, index=pd.DatetimeIndex(times).as_unit("ns"), dtype=float)


def iso_utc(t) -> str:
    """ISO 8601 UTC with Z, from an aware datetime or a naive UTC pandas timestamp."""
    if isinstance(t, datetime.datetime) and t.tzinfo is not None:
        t = t.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return t.isoformat() + "Z"
