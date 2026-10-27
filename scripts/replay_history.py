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
Replays real device history through the operator to measure how often it reports (calibration aid, not a test).

    fetch   read --days of history per device through timescale-wrapper, the same reader the operator uses,
            and cache it as parquet; needs a platform token in SENERGY_TOKEN
    replay  train on everything before the last --replay-days, play those days through infer() and the watchdog,
            print the reports per day; offline, on the cache
    recall  as replay, with anomalies injected into the replayed days; prints what was reported around each

The devices file is a JSON list of {"name", "device_id", "service_id", "column", "end" (optional ISO time)};
"column" is the path of the value in the service output, e.g. "sensor.ENERGY.Power".

    uv run python scripts/replay_history.py fetch devices.json cache/
    uv run python scripts/replay_history.py replay devices.json cache/
"""

import argparse
import collections
import datetime
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from algo import training
from algo.detectors.continuous_curve import ContCurveModel
from algo.detectors.load_curve import LoadCurveModel
from main import CustomConfig, Operator

WRAPPER_URL = "https://api.senergy.infai.org/db/v3"


class _Frame:
    def __init__(self, frame):
        self.frame = frame

    def to_pandas(self):
        return self.frame


class _Wrapped:
    def __init__(self, model):
        self.model = model

    def unwrap_python_model(self):
        return self.model


def _path(cache, device):
    return os.path.join(cache, device["name"].replace(" ", "_") + ".parquet")


def fetch(devices, cache, days):
    from operator_lib.util.helpers.ts_wrapper import get_ts_wrapper_dataset_local
    from operator_lib.util.model import InputTopic

    os.makedirs(cache, exist_ok=True)
    token = os.environ["SENERGY_TOKEN"]
    for device in devices:
        end = (datetime.datetime.fromisoformat(device["end"]) if device.get("end")
               else datetime.datetime.now(datetime.timezone.utc))
        topic = InputTopic({"name": device["service_id"].replace(":", "_"), "filterType": "DeviceId",
                            "filterValue": device["device_id"],
                            "mappings": [{"dest": "value", "source": "value." + device["column"]}]})
        for attempt in range(3):
            try:
                frame = get_ts_wrapper_dataset_local(WRAPPER_URL, token, topic, datetime.timedelta(days=days),
                                                     False, end).to_pandas()
                frame[["time", "value"]].to_parquet(_path(cache, device))
                print(device["name"], len(frame), "rows", flush=True)
                break
            except Exception as ex:
                print(device["name"], "attempt", attempt + 1, type(ex).__name__, str(ex)[:200], flush=True)


def _split(cache, device, replay_days):
    frame = pd.read_parquet(_path(cache, device))
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.dropna().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    split = frame["time"].max() - pd.Timedelta(days=replay_days)
    return frame[frame["time"] < split], frame[frame["time"] >= split].reset_index(drop=True)


def _train(train):
    model, _, _ = training.fit_anomaly_model([_Frame(train)], [{"filterType": "DeviceId", "filterValue": "d"}],
                                             train["time"].max().to_pydatetime(), True)
    return model


def _play(model, test):
    produced = []
    op = Operator()
    op.config = CustomConfig({"check_data_anomalies": True})
    state = {}
    op._now = lambda: state["wall"]  # replayed time: no backlog, the watchdog sees the gaps as they were
    op.produce = lambda report: produced.append((state["wall"], report))
    op.model = _Wrapped(model)
    reports = []
    for t, v in zip(test["time"], test["value"]):
        ts = t.to_pydatetime()
        state["wall"] = ts
        op._check_outages()
        _, results, _ = op.infer(op.model, {"value": v}, "s", "d", ts)
        reports += [(ts, r) for r in results or []]
    return reports + produced


def _kinds(reports):
    return collections.Counter(f"{r['type']}/{r.get('sub_type', '')}" for _, r in reports)


def _describe(model):
    p = model.profile("d")
    if p is None:
        return "no profile"
    text = f"type={p.device_type}"
    if isinstance(p.curve, ContCurveModel):
        text += f" threshold={p.curve.error_threshold:.4f}"
    if isinstance(p.curve, LoadCurveModel):
        text += f" runs={p.curve.n_loads} regular={p.curve.regular}"
    if p.values:
        text += f" values=[{p.values.lower():.1f}, {p.values.upper():.1f}]"
    if p.gaps:
        text += f" gap_upper={p.gaps.upper():.1f}min"
    return text


def replay(devices, cache, replay_days):
    for device in devices:
        train, test = _split(cache, device, replay_days)
        if test.empty:
            print(f"{device['name']}: nothing to replay", flush=True)
            continue
        days = (test["time"].max() - test["time"].min()).total_seconds() / 86400
        model = _train(train)
        per_day = {k: round(n / days, 2) for k, n in sorted(_kinds(_play(model, test)).items())}
        print(f"{device['name']}: {_describe(model)}\n  reports per day: {per_day}", flush=True)


def _spike(test, at):
    i = test.index[test["time"] >= at][0]
    test.loc[i, "value"] = test["value"].max() * 3
    return test, test.loc[i, "time"], test.loc[i, "time"]


def _negative(test, at):
    i = test.index[test["time"] >= at][0]
    test.loc[i, "value"] = -50.0
    return test, test.loc[i, "time"], test.loc[i, "time"]


def _outage(test, at):
    end = at + pd.Timedelta(minutes=30)
    return test[(test["time"] < at) | (test["time"] >= end)].reset_index(drop=True), at, end


def _flat(test, at, value):
    end = at + pd.Timedelta(hours=6)
    test.loc[(test["time"] >= at) & (test["time"] < end), "value"] = value
    return test, at, end


def _held_run(test, at):
    """The first run after `at` (above median + 10, the run rule) keeps its level for five hours."""
    on = test[(test["time"] >= at) & (test["value"] > float(test["value"].median()) + 10)]
    if on.empty:
        return test, at, at
    start = on["time"].iloc[0]
    end = start + pd.Timedelta(hours=5)
    test.loc[(test["time"] >= start) & (test["time"] < end), "value"] = float(on["value"].iloc[:20].median())
    return test, start, end


INJECTIONS = {
    "spike 3x max": _spike,
    "negative value": _negative,
    "outage 30 min": _outage,
    "flat at 0 for 6 h": lambda t, a: _flat(t, a, 0.0),
    "flat at 2x median for 6 h": lambda t, a: _flat(t, a, 2 * float(t["value"].median()) + 10),
    "run held for 5 h": _held_run,
}


def recall(devices, cache, replay_days):
    for device in devices:
        train, test = _split(cache, device, replay_days)
        if test.empty:
            continue
        model = _train(train)
        print(f"{device['name']}: {_describe(model)}", flush=True)
        at = test["time"].min() + pd.Timedelta(days=replay_days / 2)
        for label, inject in INJECTIONS.items():
            injected, start, end = inject(test.copy(), at)
            found = _kinds([(ts, r) for ts, r in _play(model, injected)
                            if start <= pd.Timestamp(ts) <= end + pd.Timedelta(hours=4)])
            print(f"  {label:28s} reported: {dict(found) or '-'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("fetch", "replay", "recall"))
    parser.add_argument("devices")
    parser.add_argument("cache")
    parser.add_argument("--days", type=float, default=60)
    parser.add_argument("--replay-days", type=float, default=10)
    args = parser.parse_args()
    with open(args.devices) as f:
        devices = json.load(f)
    if args.command == "fetch":
        fetch(devices, args.cache, args.days)
    elif args.command == "replay":
        replay(devices, args.cache, args.replay_days)
    else:
        recall(devices, args.cache, args.replay_days)


if __name__ == "__main__":
    main()
