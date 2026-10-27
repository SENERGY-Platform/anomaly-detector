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

Detects a value outside the bounds of the device's history. The bounds are extreme quantiles of all history values, widened by a margin; a device without a profile uses the same statistic over its last live values.
"""

import dataclasses
import typing

import numpy as np

MIN_VALUES = 30
# Margin beyond the extreme quantiles, as a share of their distance.
VALUE_MARGIN = 0.2
# Floor under that margin, as a share of the larger quantile: a constant device has no distance at all.
VALUE_MIN_MARGIN = 0.05
VALUE_QUANTILES = (0.0001, 0.9999)


@dataclasses.dataclass
class ValueStats:
    """
    The bounds come from extreme quantiles of all values, not from mean and standard deviation: a device with
    two modes (standby and heating element) or a skewed distribution has a standard deviation that puts
    mean + 5 sigma inside the upper mode.
    """
    mean: float  # for the output only
    std: float
    n: int
    q_low: float  # 0.01 % quantile
    q_high: float  # 99.99 % quantile

    def margin(self) -> float:
        return max(VALUE_MARGIN * (self.q_high - self.q_low),
                   VALUE_MIN_MARGIN * max(abs(self.q_low), abs(self.q_high)))

    def upper(self) -> float:
        return self.q_high + self.margin()

    def lower(self) -> float:
        lower = self.q_low - self.margin()
        # A quantity that never was negative, such as power: only a negative value is impossible.
        return max(lower, 0.0) if self.q_low >= 0 else lower


def value_stats(values: np.ndarray) -> ValueStats:
    """Shared by the training and the live fallback, so that both compute the same."""
    values = np.asarray(values, dtype=float)
    q_low, q_high = np.quantile(values, VALUE_QUANTILES)
    return ValueStats(float(np.mean(values)), float(np.std(values)), int(len(values)), float(q_low), float(q_high))


def _value_stats(values: np.ndarray) -> typing.Optional[ValueStats]:
    return value_stats(values) if len(values) >= MIN_VALUES else None


def check(profile, st, key, value) -> typing.Optional[dict]:
    result = None
    try:
        stats = profile.values if profile is not None and profile.values is not None else st.values.stats()
        if stats is not None:
            sub_type = threshold = None
            if value > stats.upper():
                sub_type, threshold = "high", stats.upper()
            elif value < stats.lower():
                sub_type, threshold = "low", stats.lower()
            # Once per excursion: a device stuck out of bounds would otherwise report every message.
            previous, st.extreme_side = st.extreme_side, sub_type
            if sub_type is not None and sub_type != previous:
                result = {"type": "extreme_value", "sub_type": sub_type, "value": value,
                          "mean": round(stats.mean, 2), "threshold": round(threshold, 2), "device_id": key}
    finally:
        st.values.add(value)
    return result
