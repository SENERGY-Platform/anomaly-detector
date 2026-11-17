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

"""
The statistics a device is judged by: the range of its values and the longest usual silence between its messages.
Both come from the history of the device, or from its last live values and gaps while it has none.
"""

__all__ = ("ValueStats", "GapStats", "value_stats", "gap_stats", "SampleBuffer")

import collections
import dataclasses
import typing

import numpy as np

# A statistic needs at least this many values or gaps.
MIN_SAMPLES = 30

# The bounds of the values are their range without the VALUE_IGNORED most extreme values on each side, but at most
# one in VALUE_IGNORED_PER values, widened by VALUE_MARGIN of that range on both sides and at least by
# VALUE_MIN_MARGIN of its larger end, since a constant device has no range at all.
# A number of values rather than a quantile: 50 days hold about 1,200 values of a device that sends every hour and
# 72,000 of one that sends every minute, and a quantile such as 99.99 % is the maximum of the one, so that a single
# outlier hides itself for 50 days, and about the 8th largest value of the other, too close for a margin of 50 %.
# Measured on public data in docs/extreme-value-bounds-and-hysteresis.md.
# The range rather than mean and standard deviation: a device with two modes, such as standby and a heating element,
# has its mean + 5 sigma inside the upper mode.
VALUE_IGNORED = 5
VALUE_IGNORED_PER = 200
VALUE_MARGIN = 0.5
VALUE_MIN_MARGIN = 0.05

# A device is silent after GAP_FACTOR times the 99.9 % quantile of its gaps, and never before GAP_MIN_MINUTES: a
# device that sends every few seconds pauses for minutes at idle.
GAP_QUANTILE = 0.999
GAP_FACTOR = 3
GAP_MIN_MINUTES = 5

# A device without history is judged by its last FALLBACK_SAMPLES values and gaps. Their quantiles cost about a
# millisecond, so they are recomputed every STATS_REFRESH samples, but with every sample while there are fewer than
# STATS_REFRESH_ALWAYS_BELOW: a young statistic frozen for 100 samples would report every value of a new, normal level.
FALLBACK_SAMPLES = 10_000
STATS_REFRESH = 100
STATS_REFRESH_ALWAYS_BELOW = 1_000


@dataclasses.dataclass(frozen=True)
class ValueStats:
    mean: float  # for the output only
    q_low: float
    q_high: float

    def margin(self) -> float:
        return max(VALUE_MARGIN * (self.q_high - self.q_low),
                   VALUE_MIN_MARGIN * max(abs(self.q_low), abs(self.q_high)))

    def upper(self) -> float:
        return self.q_high + self.margin()

    def lower(self) -> float:
        lower = self.q_low - self.margin()
        # A quantity that never was negative, such as power: only a negative value is impossible.
        return max(lower, 0.0) if self.q_low >= 0 else lower


@dataclasses.dataclass(frozen=True)
class GapStats:
    mean_min: float  # for the output only
    q999_min: float

    def upper(self) -> float:
        """The longest silence in minutes that is still usual."""
        return max(GAP_MIN_MINUTES, GAP_FACTOR * self.q999_min)


def value_stats(values) -> typing.Optional[ValueStats]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < MIN_SAMPLES:
        return None
    ignored = min(VALUE_IGNORED, n // VALUE_IGNORED_PER)
    ends = np.partition(values, (ignored, n - 1 - ignored))
    return ValueStats(float(np.mean(values)), float(ends[ignored]), float(ends[n - 1 - ignored]))


def gap_stats(gaps_min) -> typing.Optional[GapStats]:
    gaps_min = np.asarray(gaps_min, dtype=float)
    if len(gaps_min) < MIN_SAMPLES:
        return None
    return GapStats(float(np.mean(gaps_min)), float(np.quantile(gaps_min, GAP_QUANTILE)))


class SampleBuffer:
    """The last FALLBACK_SAMPLES values (or gaps) of a device and their statistic, from value_stats or gap_stats."""

    def __init__(self, make_stats: typing.Callable[[np.ndarray], typing.Any]):
        self._make_stats = make_stats
        self._samples: typing.Deque[float] = collections.deque(maxlen=FALLBACK_SAMPLES)
        self._stats = None
        self._added = 0  # since the statistic was computed

    def add(self, x: float):
        self._samples.append(x)
        self._added += 1

    def stats(self):
        refresh = 1 if len(self._samples) < STATS_REFRESH_ALWAYS_BELOW else STATS_REFRESH
        if self._added >= refresh:
            self._stats = self._make_stats(np.fromiter(self._samples, dtype=float, count=len(self._samples)))
            self._added = 0
        return self._stats
