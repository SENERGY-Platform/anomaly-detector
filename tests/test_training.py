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

import datetime
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from algo.detectors import extreme_value, receive_time
from algo.detectors.extreme_value import ValueStats
from algo.detectors.receive_time import GapStats
from algo.model import SCHEMA_VERSION
from algo.training import fit_anomaly_model

NOW = datetime.datetime(2026, 1, 10, tzinfo=datetime.timezone.utc)
START = pd.Timestamp("2026-01-01")


class FakeDataset:
    def __init__(self, frame):
        self.frame = frame

    def to_pandas(self):
        return self.frame.copy()


class UnreadableDataset:
    def to_pandas(self):
        raise RuntimeError("Failed to get allocated resources")


def day_curve(days, step_s=30, seed=0, start=START):
    """A daily cycle around 100 with noise, one value per step_s."""
    n = int(days * 86400 / step_s)
    t = start + pd.to_timedelta(np.arange(n) * step_s, unit="s")
    rng = np.random.RandomState(seed)
    values = 100 + 50 * np.sin(2 * np.pi * np.arange(n) * step_s / 86400) + rng.normal(0, 2, n)
    return pd.Series(values, index=t)


def to_frame(series, **extra):
    frame = pd.DataFrame({"time": series.index, "value": series.to_numpy()})
    for k, v in extra.items():
        frame[k] = v
    return frame



def fit(datasets, meta=None):
    meta = meta if meta is not None else [{"filterType": "DeviceId", "filterValue": "d1"}] * len(datasets or [])
    return fit_anomaly_model(datasets, meta, NOW)


class Profile(unittest.TestCase):
    def test_profile(self):
        model, notes, failed = fit([FakeDataset(to_frame(day_curve(4)))])
        self.assertFalse(failed)
        self.assertEqual(model.schema_version, SCHEMA_VERSION)
        self.assertEqual(model.trained_until, NOW)
        self.assertEqual(model.predict(None, None), ["d1"])
        p = model.profile("d1")
        self.assertIsInstance(p.values, ValueStats)
        self.assertIsInstance(p.gaps, GapStats)
        self.assertAlmostEqual(p.gaps.mean_min, 0.5, places=3)
        self.assertAlmostEqual(p.values.mean, 100, delta=3)
        self.assertGreater(p.values.upper(), p.values.mean)
        self.assertLess(p.values.lower(), p.values.mean)
        self.assertEqual(len(notes), 1)


class Reading(unittest.TestCase):
    def test_unordered_shards_mixed_timezones_strings_missing_values(self):
        series = day_curve(1, step_s=60)
        frame = to_frame(series)
        frame["value"] = frame["value"].astype(str)
        frame.loc[5, "value"] = None
        frame.loc[6, "value"] = "n/a"
        frame.loc[7, "time"] = None
        shards = [frame.iloc[i::4] for i in range(4)]
        shards = [shards[2], shards[0], shards[3], shards[1]]
        aware = shards[1].copy()
        aware["time"] = pd.to_datetime(aware["time"]).dt.tz_localize("UTC")
        shards[1] = aware
        # The shards are one topic; the reader concatenates them, here in the wrong order.
        model, notes, failed = fit([FakeDataset(pd.concat(shards))])
        p = model.profile("d1")
        self.assertFalse(failed)
        self.assertEqual(p.values.n, 1440 - 3)
        # One gap is bridged by the rows without a value
        self.assertAlmostEqual(p.gaps.mean_min, (1440 - 4) / (1440 - 3 - 1) * 1.0, delta=0.01)

    def test_naive_and_aware_agree(self):
        series = day_curve(1, step_s=60)
        naive = fit([FakeDataset(to_frame(series))])[0].profile("d1")
        aware_frame = to_frame(series)
        aware_frame["time"] = aware_frame["time"].dt.tz_localize("Europe/Berlin")
        aware = fit([FakeDataset(aware_frame)])[0].profile("d1")
        self.assertAlmostEqual(naive.values.mean, aware.values.mean)
        self.assertEqual(naive.gaps.n, aware.gaps.n)

    def test_group_by_original_input_ids_for_operator_topics(self):
        a, b = day_curve(1, step_s=60), day_curve(1, step_s=60, seed=1) + 50
        frame = pd.concat([to_frame(a, original_input_ids="devA"), to_frame(b, original_input_ids="devB")])
        meta = [{"filterType": "OperatorId", "filterValue": "op1"}]
        model, _, _ = fit([FakeDataset(frame)], meta)
        self.assertEqual(sorted(model.devices), ["devA", "devB"])
        self.assertGreater(model.profile("devB").values.mean, model.profile("devA").values.mean + 40)

    def test_device_id_topic_ignores_the_column(self):
        frame = to_frame(day_curve(1, step_s=60), original_input_ids="other")
        model, _, _ = fit([FakeDataset(frame)], [{"filterType": "DeviceId", "filterValue": "d1"}])
        self.assertEqual(list(model.devices), ["d1"])

    def test_device_topic_with_several_devices_groups_by_device_column(self):
        a, b = day_curve(1, step_s=60), day_curve(1, step_s=60, seed=1) + 50
        frame = pd.concat([to_frame(a, device_id="d1"), to_frame(b, device_id="d2")])
        model, _, _ = fit([FakeDataset(frame)], [{"filterType": "DeviceId", "filterValue": "d1,d2"}])
        self.assertEqual(sorted(model.devices), ["d1", "d2"])
        self.assertGreater(model.profile("d2").values.mean, model.profile("d1").values.mean + 40)

    def test_device_filter_value_is_normalised_to_one_device_key(self):
        frame = to_frame(day_curve(1, step_s=60))
        for raw in ("a, ", " a", "a", "a,,"):
            model, _, _ = fit([FakeDataset(frame)], [{"filterType": "DeviceId", "filterValue": raw}])
            self.assertEqual(list(model.devices), ["a"], raw)

    def test_several_devices_without_a_column_keep_the_stripped_raw_value(self):
        model, _, _ = fit([FakeDataset(to_frame(day_curve(1, step_s=60)))],
                          [{"filterType": "DeviceId", "filterValue": " a, b "}])
        self.assertEqual(list(model.devices), ["a, b"])

    def test_two_topics_with_the_same_device_are_merged(self):
        series = day_curve(1, step_s=60)
        first, second = series[:720], series[720:]
        meta = [{"filterType": "DeviceId", "filterValue": "d1"}, {"filterType": "DeviceId", "filterValue": "d1"}]
        model, _, _ = fit([FakeDataset(to_frame(second)), FakeDataset(to_frame(first))], meta)
        self.assertEqual(list(model.devices), ["d1"])
        self.assertEqual(model.profile("d1").values.n, 1440)
        self.assertAlmostEqual(model.profile("d1").gaps.mean_min, 1.0)

    def test_frame_without_columns_is_skipped(self):
        model, notes, failed = fit([FakeDataset(pd.DataFrame({"x": [1]}))])
        self.assertEqual(model.devices, {})
        self.assertFalse(failed)

    def test_empty_history(self):
        for datasets in (None, [FakeDataset(pd.DataFrame(columns=["time", "value"]))]):
            model, _, failed = fit(datasets, meta=[{"filterType": "DeviceId", "filterValue": "d1"}])
            self.assertEqual(model.devices, {})
            self.assertFalse(failed)
            self.assertEqual(model.trained_until, NOW)

    def test_unreadable_history(self):
        good = FakeDataset(to_frame(day_curve(1, step_s=60)))
        model, notes, failed = fit([UnreadableDataset(), good])
        self.assertTrue(failed)
        self.assertEqual(list(model.devices), ["d1"])
        self.assertTrue(any("not readable" in n for n in notes))
        model, _, failed = fit([UnreadableDataset()])
        self.assertTrue(failed)
        self.assertEqual(model.devices, {})


class Stats(unittest.TestCase):
    def test_value_bounds_are_quantiles_with_a_margin(self):
        stats = ValueStats(mean=10.0, std=3.0, n=100, q_low=-10.0, q_high=90.0)
        self.assertEqual((stats.lower(), stats.upper()), (-30.0, 110.0))

    def test_lower_bound_of_a_never_negative_quantity_is_zero(self):
        stats = ValueStats(mean=10.0, std=3.0, n=100, q_low=0.0, q_high=100.0)
        self.assertEqual((stats.lower(), stats.upper()), (0.0, 120.0))

    def test_gap_upper_has_a_floor_and_a_factor(self):
        self.assertEqual(GapStats(0.5, 100, 0.5).upper(), 5)
        self.assertEqual(GapStats(1.0, 100, 10.0).upper(), 30.0)

    def test_the_training_and_the_live_fallback_compute_the_same(self):
        from algo import live
        values = np.random.RandomState(3).gamma(2.0, 30.0, 500)
        buffer = live.SampleBuffer(extreme_value.value_stats, extreme_value.MIN_VALUES)
        for v in values:
            buffer.add(float(v))
        self.assertEqual(buffer.stats(), extreme_value._value_stats(values))
        index = pd.DatetimeIndex(pd.Timestamp("2026-01-01") + pd.to_timedelta(np.cumsum(values), unit="s"))
        gaps = live.SampleBuffer(receive_time.gap_stats, receive_time.MIN_GAPS)
        for g in np.diff(index.to_numpy()) / np.timedelta64(1, "m"):
            gaps.add(float(g))
        self.assertEqual(gaps.stats(), receive_time._gap_stats(index))


class BimodalDevice(unittest.TestCase):
    """A pump at about 40 W and a heating element at up to 1950 W, 0 W at rest, as on a flushing cycle."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(4)
        values = []
        for _ in range(60):
            values += [0.0] * 120
            values += list(40 + rng.normal(0, 2, 30))
            values += list(rng.uniform(1700, 1950, 20))
        index = START + pd.to_timedelta(np.arange(len(values)), unit="min")
        cls.model, _, _ = fit([FakeDataset(to_frame(pd.Series(values, index=index)))])
        cls.profile = cls.model.profile("d1")

    def test_bounds_hold_the_upper_mode_and_leave_standby_alone(self):
        v = self.profile.values
        self.assertGreaterEqual(v.upper(), 1950)
        self.assertLess(v.upper(), 3000)
        self.assertEqual(v.lower(), 0.0)  # power is never negative


class Minimums(unittest.TestCase):
    def test_too_little_data_leaves_parts_empty(self):
        short = day_curve(0.01, step_s=60)[:20]
        p = fit([FakeDataset(to_frame(short))])[0].profile("d1")
        self.assertIsNone(p.values)
        self.assertIsNone(p.gaps)


if __name__ == "__main__":
    unittest.main()
