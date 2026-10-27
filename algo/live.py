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

__all__ = ("SampleBuffer", "DeviceState", "BUFFER_MINUTES", "FALLBACK_SAMPLES", "STATS_REFRESH")

import collections
import datetime
import typing

import numpy as np

from . import preprocessing
from .detectors.extreme_value import MIN_VALUES, value_stats
from .detectors.load_curve import RunTracker
from .detectors.receive_time import MIN_GAPS, gap_stats


BUFFER_MINUTES = 240

# The statistic of a device without a profile comes from its last values and gaps, as many as the history gives.
FALLBACK_SAMPLES = 10_000
# The quantiles of 10_000 values cost about a millisecond; every message would pay for it twice. A statistic of
# 10_000 values moves little within 100 new ones.
STATS_REFRESH = 100
# Below this many samples every add recomputes: a young statistic frozen for 100 messages reports every message
# of a new, normal level.
STATS_REFRESH_ALWAYS_BELOW = 1_000


class SampleBuffer:
    """The last FALLBACK_SAMPLES values (or gaps) of a device and their statistic, recomputed every STATS_REFRESH adds."""

    def __init__(self, make_stats: typing.Callable[[np.ndarray], typing.Any], min_n: int):
        self._make_stats = make_stats
        self._min_n = min_n
        self.samples: typing.Deque[float] = collections.deque(maxlen=FALLBACK_SAMPLES)
        self._stats = None
        self._since = 0

    def add(self, x: float):
        self.samples.append(x)
        self._since += 1

    def stats(self):
        """The statistic, once min_n samples exist; at most STATS_REFRESH adds old."""
        if len(self.samples) < self._min_n:
            return None
        refresh = 1 if len(self.samples) < STATS_REFRESH_ALWAYS_BELOW else STATS_REFRESH
        if self._stats is None or self._since >= refresh:
            self._stats = self._make_stats(np.fromiter(self.samples, dtype=float, count=len(self.samples)))
            self._since = 0
        return self._stats


class DeviceState:
    """
    The runtime state of one device key. `gaps`, `last_arrival` and `armed` are shared with the watchdog
    thread and only touched under the operator's lock; the rest belongs to infer().
    """

    def __init__(self):
        self.values = SampleBuffer(value_stats, MIN_VALUES)
        self.gaps = SampleBuffer(gap_stats, MIN_GAPS)
        self.last_arrival: typing.Optional[datetime.datetime] = None  # Kafka time of the last message
        self.armed = False  # whether a missing message may be reported
        self.buffer: typing.Deque[preprocessing.Sample] = collections.deque()
        self.last_check_minute: typing.Optional[datetime.datetime] = None
        self.last_hit: typing.Optional[datetime.datetime] = None
        self.curve_locked_until: typing.Optional[datetime.datetime] = None
        self.extreme_side: typing.Optional[str] = None  # "high" or "low" while values are out of bounds
        self.run = RunTracker()

    def push(self, ts: datetime.datetime, value: float):
        self.buffer.append((ts, value))
        limit = ts - datetime.timedelta(minutes=BUFFER_MINUTES)
        while self.buffer and self.buffer[0][0] < limit:
            self.buffer.popleft()
