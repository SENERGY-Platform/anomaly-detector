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

Detects a gap between messages that is too long for the device, once per outage. The bound comes from the gaps in the history.
"""

import dataclasses
import typing

import numpy as np
import pandas as pd

MIN_GAPS = 30
# A device that sends every few seconds but pauses for minutes at idle: the bound is a multiple of the 99.9 %
# quantile of its gaps, never below this.
GAP_MIN_UPPER_MIN = 5  # minutes
GAP_UPPER_FACTOR = 3
GAP_QUANTILE = 0.999


@dataclasses.dataclass
class GapStats:
    mean_min: float  # for the output only
    n: int
    q999_min: float  # 99.9 % quantile of the gaps

    def upper(self) -> float:
        return max(GAP_MIN_UPPER_MIN, GAP_UPPER_FACTOR * self.q999_min)


def gap_stats(gaps_min: np.ndarray) -> GapStats:
    gaps_min = np.asarray(gaps_min, dtype=float)
    return GapStats(float(np.mean(gaps_min)), int(len(gaps_min)), float(np.quantile(gaps_min, GAP_QUANTILE)))


def _gap_stats(index: pd.DatetimeIndex) -> typing.Optional[GapStats]:
    gaps = np.diff(index.to_numpy()) / np.timedelta64(1, "m")
    return gap_stats(gaps) if len(gaps) >= MIN_GAPS else None


def arrive(st, ts):
    """Records a message; one out of order or a duplicate is neither a gap nor a newer arrival."""
    prev = st.last_arrival
    if prev is not None and ts <= prev:
        return
    if prev is not None:
        st.gaps.add((ts - prev).total_seconds() / 60)
    st.last_arrival = ts
    st.armed = True


def outage(profile, st, key, now) -> typing.Optional[dict]:
    if not st.armed or st.last_arrival is None:
        return None
    stats = profile.gaps if profile is not None and profile.gaps is not None else st.gaps.stats()
    if stats is None:
        return None
    minutes = (now - st.last_arrival).total_seconds() / 60
    if minutes > stats.upper():
        st.armed = False  # once per outage, armed again by the next message
        return {"type": "time", "sub_type": "high", "value": round(minutes, 2),
                "mean": round(stats.mean_min, 2), "threshold": round(stats.upper(), 2), "device_id": key}
    return None
