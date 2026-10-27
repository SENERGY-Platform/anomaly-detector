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

__all__ = ("_read_history",)

import typing

import numpy as np
import pandas as pd
import ray


def _read_frame(ds) -> pd.DataFrame:
    frame = (ray.get(ds) if isinstance(ds, ray.ObjectRef) else ds).to_pandas()
    return frame


def _prepare(frame: pd.DataFrame) -> typing.Optional[pd.DataFrame]:
    """Parsed time (naive UTC) and numeric value, sorted by time; None without those columns."""
    if "time" not in frame.columns or "value" not in frame.columns:
        return None
    frame = frame.copy()
    # Remote Timescale delivers aware times, Kafka naive UTC.
    frame["time"] = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True, errors="coerce")).tz_localize(None).as_unit("ns")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[frame["time"].notna() & np.isfinite(frame["value"].astype(float))]
    # read_sql over several shards guarantees no overall order.
    return frame.sort_values("time", kind="stable")


def _read_history(datasets, topic_meta, notes: typing.List[str]):
    """Series per device key, and whether a requested history could not be read."""
    if datasets is None:
        notes.append("no history requested")
        return {}, False
    failed = False
    parts: typing.Dict[str, typing.List[pd.Series]] = {}
    for i, ds in enumerate(datasets):
        try:
            frame = _read_frame(ds)
        except Exception as ex:
            failed = True
            notes.append(f"history of topic {i} not readable: {type(ex).__name__}: {ex}")
            continue
        frame = _prepare(frame)
        if frame is None:
            notes.append(f"history of topic {i} has no time or value column")
            continue
        if frame.empty:
            continue
        meta = topic_meta[i] if i < len(topic_meta) else {}
        if meta.get("filterType") == "DeviceId" and "device_id" in frame.columns:
            # A topic naming several devices ("a,b") comes back with a device column per row.
            keyed = frame[frame["device_id"].notna()]
            groups = [(str(k), g) for k, g in keyed.groupby(keyed["device_id"].astype(str))]
        elif meta.get("filterType") != "DeviceId" and "original_input_ids" in frame.columns:
            keyed = frame[frame["original_input_ids"].notna()]
            groups = [(str(k), g) for k, g in keyed.groupby(keyed["original_input_ids"].astype(str))]
        elif meta.get("filterType") == "DeviceId":
            # One device, or several without a device column; the key as the runtime has it, without the
            # blanks a hand-written filter value may carry.
            raw = str(meta.get("filterValue"))
            devices = [d for d in (part.strip() for part in raw.split(",")) if d]
            groups = [(devices[0] if len(devices) == 1 else raw.strip(), frame)]
        else:
            groups = [(str(meta.get("filterValue")), frame)]
        for key, group in groups:
            parts.setdefault(key, []).append(pd.Series(group["value"].to_numpy(float), index=pd.DatetimeIndex(group["time"])))
    series = {}
    for key, ps in parts.items():
        merged = pd.concat(ps).sort_index(kind="stable")
        series[key] = merged[~merged.index.duplicated(keep="first")]
    return series, failed
