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

import dataclasses
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from operator_lib.util import clock

import main
from main import HISTORY_RETRIES, HISTORY_RETRY_S, LAG_LIMIT_S, CustomConfig, Operator
from algo import live
from algo.live import FALLBACK_SAMPLES, STATS_REFRESH
from algo.detectors.continuous_curve import CURVE_LOCKOUT, Autoencoder, ContCurveModel
from algo.detectors.extreme_value import MIN_VALUES, ValueStats, value_stats
from algo.detectors.load_curve import LoadCurveModel, RunTracker, fit_load_curve
from algo.detectors.receive_time import MIN_GAPS, GapStats
from algo.model import SCHEMA_VERSION, WINDOW, AnomalyModel, DeviceProfile
from algo.training import fit_anomaly_model
from tests.test_training import START, day_curve, to_frame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTC = datetime.timezone.utc
T0 = datetime.datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
KEY = "dev"
MINUTE = datetime.timedelta(minutes=1)


def at(minutes: float) -> datetime.datetime:
    return T0 + datetime.timedelta(minutes=minutes)


class FakePyFuncModel:
    """What mlflow.pyfunc.load_model returns: the python model behind unwrap_python_model()."""

    def __init__(self, python_model):
        self.python_model = python_model

    def unwrap_python_model(self):
        return self.python_model


class FakeDataset:
    def __init__(self, frame):
        self.frame = frame

    def to_pandas(self):
        return self.frame.copy()



class UnreadableDataset:
    def to_pandas(self):
        raise RuntimeError("Failed to get allocated resources")


@dataclasses.dataclass
class StubCurve(ContCurveModel):
    """Error is the mean deviation from the window mean: zero for a flat window, large for a deformed one."""

    def reconstruct(self, window):
        window = np.asarray(window, dtype=float)
        recon = np.full(WINDOW, window.mean())
        return recon, float(np.abs(window - recon).mean() / self.training_max)


@dataclasses.dataclass
class RaisingCurve(ContCurveModel):
    def reconstruct(self, window):
        raise RuntimeError("broken network")


def stub_curve(cls=StubCurve, threshold=0.05):
    return cls(state_dict={}, training_max=100.0, error_threshold=threshold)


def profile(values=ValueStats(100.0, 10.0, 1000, 65.0, 135.0), gaps=GapStats(10.0, 1.0, 1000, 12.0), curve=None,
            device_type="cont_device", baseline=100.0):
    return DeviceProfile(device_type, baseline, values, gaps, curve)


def anomaly_model(devices, trained_until=T0, schema_version=SCHEMA_VERSION):
    return AnomalyModel(schema_version, trained_until, devices)


def load_curve():
    """Regular runs of 36 to 44 minutes at 300."""
    return fit_load_curve([np.full(int(n), 300.0) for n in np.linspace(36, 44, 40)])


class OperatorTestCase(unittest.TestCase):
    def setUp(self):
        clock.set_fixed(T0)
        self.addCleanup(clock.set_fixed, None)
        self.wall = T0
        self.produced = []

    def operator(self, devices=None, model=None, **config):
        op = Operator()
        op.config = CustomConfig(dict(config))
        op._now = lambda: self.wall
        op.produce = self.produced.append
        if model is None and devices is not None:
            model = anomaly_model(devices)
        op.model = FakePyFuncModel(model) if isinstance(model, AnomalyModel) else model
        return op

    def infer(self, op, value, ts, device_id=KEY, **extra):
        data = {"value": value, **extra}
        _, results, new_model = op.infer(op.model, data, "s", device_id, ts)
        self.assertIsNone(new_model)
        return results


class TestRun(OperatorTestCase):
    def test_one_value_through_run_gives_no_finding(self):
        op = self.operator({KEY: profile()})
        self.assertFalse(op.need_retraining(op.model))
        out = op.run({"value": 100.0, "original_input_ids": "x"}, "s", KEY, T0)
        self.assertEqual(out, (T0, None))

    def test_two_findings_in_one_infer_give_a_list_of_two(self):
        op = self.operator({KEY: profile()})
        self.assertIsNone(self.infer(op, 100.0, at(0)))
        self.wall = at(1)
        results = op.run({"value": 1000.0}, "s", KEY, at(1))[1]
        self.assertEqual(sorted(r["type"] for r in results), ["extreme_value", "time"])

    def test_device_key_falls_back_to_the_original_input_id(self):
        op = self.operator({"orig": profile()})
        results = self.infer(op, 1000.0, T0, device_id="", original_input_ids="orig")
        self.assertEqual(results[0]["device_id"], "orig")
        self.assertEqual(results[0]["threshold"], 149.0)

    def test_message_without_value_does_not_raise(self):
        op = self.operator({KEY: profile()})
        _, results, _ = op.infer(op.model, {"value": None}, "s", KEY, T0)
        self.assertIsNone(results)


class TestExtreme(OperatorTestCase):
    def test_spike_gives_extreme_value_with_the_upper_limit_as_threshold(self):
        op = self.operator({KEY: profile()})
        (result,) = self.infer(op, 180.0, T0)
        self.assertEqual(result, {"type": "extreme_value", "sub_type": "high", "value": 180.0, "mean": 100.0,
                                  "threshold": 149.0, "device_id": KEY})

    def test_low_value_gives_the_lower_limit_as_threshold(self):
        op = self.operator({KEY: profile()})
        (result,) = self.infer(op, 20.0, T0)
        self.assertEqual((result["sub_type"], result["threshold"]), ("low", 51.0))

    def test_one_report_per_excursion(self):
        op = self.operator({KEY: profile()}, check_receive_time_outlier=False)
        reports = [self.infer(op, 180.0, at(i)) for i in range(10)]
        self.assertEqual(sum(r is not None for r in reports), 1)
        self.assertIsNone(self.infer(op, 100.0, at(10)))
        self.assertIsNotNone(self.infer(op, 180.0, at(11)))
        self.assertIsNotNone(self.infer(op, 20.0, at(12)))  # the other side is a new excursion

    def test_flag_off_gives_nothing(self):
        op = self.operator({KEY: profile()}, check_data_extreme_outlier=False)
        self.assertIsNone(self.infer(op, 180.0, T0))

    def test_load_device_in_standby_reports_no_low(self):
        # Standby at the 0.01 % quantile puts the lower bound below zero.
        load = profile(ValueStats(300.0, 20.0, 100, 5.0, 800.0), None, None, "load_device", 5.0)
        op = self.operator({KEY: load})
        self.assertIsNone(self.infer(op, 5.0, T0))
        self.assertIsNone(self.infer(op, 0.0, at(1)))

    def test_load_device_is_checked_like_any_other_in_a_run(self):
        load = profile(ValueStats(300.0, 20.0, 100, 250.0, 350.0), None, None, "load_device", 5.0)
        op = self.operator({KEY: load})
        self.assertIsNone(self.infer(op, 250.0, at(0)))
        (result,) = self.infer(op, 100.0, at(1))
        self.assertEqual((result["sub_type"], result["threshold"]), ("low", 230.0))

    def test_new_device_reports_through_the_running_statistic_after_min_values(self):
        op = self.operator(None, check_receive_time_outlier=False)
        values = [99.0 + 2 * (i % 2) for i in range(MIN_VALUES)]
        for i, v in enumerate(values):
            self.assertIsNone(self.infer(op, v, at(i)))
        (result,) = self.infer(op, 120.0, at(MIN_VALUES))
        expected = value_stats(np.array(values))  # the calculation of the training
        self.assertEqual((result["sub_type"], result["threshold"], result["mean"]),
                         ("high", round(expected.upper(), 2), 100.0))
        self.assertAlmostEqual(result["threshold"], 106.05, places=2)  # floor: 5 % of 101

    def test_new_device_reports_a_new_normal_level_once(self):
        op = self.operator(None, check_receive_time_outlier=False)
        for i in range(MIN_VALUES):
            self.infer(op, 200.0 + (i % 3 - 1), at(i))
        reports = [self.infer(op, 230.0, at(MIN_VALUES + i)) for i in range(20)]
        self.assertEqual(sum(r is not None for r in reports), 1)

    def test_the_fallback_buffer_holds_the_last_samples_only(self):
        op = self.operator(None)
        st = None
        for i in range(FALLBACK_SAMPLES + 50):
            self.infer(op, 100.0 + (i % 7), at(i / 60))
            st = op._live[KEY]
        self.assertEqual(len(st.values.samples), FALLBACK_SAMPLES)
        self.assertLessEqual(st.values.stats().n, FALLBACK_SAMPLES)
        self.assertEqual(len(st.gaps.samples), FALLBACK_SAMPLES)

    def test_a_young_fallback_statistic_follows_every_value(self):
        buffer = live.SampleBuffer(value_stats, MIN_VALUES)
        for i in range(MIN_VALUES):
            buffer.add(float(i))
        self.assertEqual(buffer.stats().n, MIN_VALUES)
        buffer.add(1000.0)
        self.assertEqual(buffer.stats().n, MIN_VALUES + 1)

    def test_an_old_fallback_statistic_is_refreshed_every_stats_refresh_values(self):
        buffer = live.SampleBuffer(value_stats, MIN_VALUES)
        for i in range(live.STATS_REFRESH_ALWAYS_BELOW):
            buffer.add(float(i % 30))
        first = buffer.stats()
        for i in range(STATS_REFRESH - 1):
            buffer.add(1000.0)
        self.assertIs(buffer.stats(), first)
        buffer.add(1000.0)
        self.assertEqual(buffer.stats().n, live.STATS_REFRESH_ALWAYS_BELOW + STATS_REFRESH)

    def test_new_device_reports_nothing_before_min_values(self):
        op = self.operator(None, check_receive_time_outlier=False)
        for i in range(MIN_VALUES - 1):
            self.infer(op, 100.0, at(i))
        self.assertIsNone(self.infer(op, 5000.0, at(MIN_VALUES)))

    def test_the_profile_is_never_changed_by_a_message(self):
        p = profile()
        op = self.operator({KEY: p})
        self.infer(op, 180.0, T0)
        self.assertEqual(p.values, ValueStats(100.0, 10.0, 1000, 65.0, 135.0))


class TestRunTracker(unittest.TestCase):
    def test_follows_the_training_rule_sample_by_sample(self):
        from algo import preprocessing
        rng = np.random.RandomState(1)
        level = np.concatenate([np.full(rng.randint(5, 60), rng.choice([5.0, 5.5, 40.0, 300.0])) for _ in range(60)])
        values = level + rng.normal(0, 3, len(level)) * (level > 20)
        index = pd.date_range("2026-03-10", periods=len(values), freq="min")
        expected = preprocessing.active_mask(pd.Series(values, index=index), 5.0).to_numpy()
        tracker = RunTracker()
        got = [tracker.update(t.to_pydatetime().replace(tzinfo=UTC), float(v), 5.0)[0] for t, v in zip(index, values)]
        # The training marks the sample before a start as part of the run, which a live check cannot know yet.
        starts = np.flatnonzero(np.diff(np.append(False, got).astype(int)) == 1)
        expected = expected.copy()
        expected[starts[starts > 0] - 1] = False
        self.assertTrue(expected.any())
        self.assertEqual(got, list(expected))


class TestGap(OperatorTestCase):
    def test_too_short_gap_gives_time_low(self):
        op = self.operator({KEY: profile()})
        self.infer(op, 100.0, at(0))
        (result,) = self.infer(op, 100.0, at(1))
        self.assertEqual(result, {"type": "time", "sub_type": "low", "value": 1.0, "mean": 10.0, "threshold": 5.0,
                                  "device_id": KEY})

    def test_regular_gap_gives_nothing(self):
        op = self.operator({KEY: profile()})
        self.infer(op, 100.0, at(0))
        self.assertIsNone(self.infer(op, 100.0, at(10)))

    def test_duplicate_timestamp_is_no_gap(self):
        regular = profile(gaps=GapStats(1.0, 0.001, 1000, 1.0))  # lower bound just below a minute
        op = self.operator({KEY: regular}, check_data_extreme_outlier=False)
        self.infer(op, 100.0, at(0))
        self.assertIsNone(self.infer(op, 100.0, at(1)))
        self.assertIsNone(self.infer(op, 100.0, at(1)))

    def test_out_of_order_message_is_no_gap(self):
        op = self.operator({KEY: profile()})
        self.infer(op, 100.0, at(10))
        self.assertIsNone(self.infer(op, 100.0, at(5)))
        self.assertEqual(op._live[KEY].last_arrival, at(10))

    def test_new_device_uses_the_running_gap_statistic(self):
        op = self.operator(None, check_data_extreme_outlier=False)
        for i in range(MIN_GAPS + 1):
            self.assertIsNone(self.infer(op, 100.0, at(10 * i)))
        # Ten minutes every time: std 0, so anything shorter is too short.
        (result,) = self.infer(op, 100.0, at(10 * (MIN_GAPS + 1) - 5))
        self.assertEqual((result["type"], result["sub_type"], result["threshold"]), ("time", "low", 10.0))

    def test_flag_off_gives_nothing(self):
        op = self.operator({KEY: profile()}, check_receive_time_outlier=False)
        self.infer(op, 100.0, at(0))
        self.assertIsNone(self.infer(op, 100.0, at(1)))


class TestContCurve(OperatorTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.real_curve = ContCurveModel(
            {k: v.detach().clone() for k, v in Autoencoder().state_dict().items()}, 200.0, 0.0)

    def operator_with(self, curve, spike=True):
        """250 buffered minutes, flat at 100 except a spike of 20 minutes inside the window."""
        op = self.operator({KEY: profile(curve=curve)}, check_data_anomalies=True, check_receive_time_outlier=False,
                           check_data_extreme_outlier=False)
        st = op._live.setdefault(KEY, main.live.DeviceState())
        for k in range(250):
            flat = not (spike and 150 <= k <= 170)
            st.push(at(k), 100.0 if flat else 300.0)
        return op, st

    def test_first_hit_alone_reports_nothing(self):
        op, st = self.operator_with(stub_curve())
        self.assertIsNone(self.infer(op, 100.0, at(250)))
        self.assertEqual(st.last_hit, at(250))

    def test_second_hit_within_30_minutes_reports_a_parsable_curve(self):
        op, _ = self.operator_with(stub_curve())
        self.infer(op, 100.0, at(250))
        (result,) = self.infer(op, 100.0, at(251))
        self.assertEqual((result["type"], result["sub_type"], result["device_id"]),
                         ("curve_anomaly", "continous_device", KEY))
        curves = json.loads(result["original_reconstructed_curves"])
        self.assertEqual(len(curves), WINDOW)
        self.assertTrue(all(len(c) == 3 and isinstance(c[0], int) for c in curves))
        self.assertTrue(all(isinstance(x, float) for c in curves for x in c[1:]))
        # Millisecond stamps of consecutive minutes.
        self.assertEqual({b[0] - a[0] for a, b in zip(curves, curves[1:])}, {60000})
        self.assertEqual(curves[0][0], pd.Timestamp(result["start_time"].rstrip("Z")).value // 10**6)
        self.assertEqual(result["value"], result["start_time"])
        self.assertTrue(result["start_time"].endswith("Z") and result["end_time"].endswith("Z"))
        # The centred smoothing leaves the last 14 minutes empty, so the window ends 14 minutes before the last one.
        self.assertEqual(pd.Timestamp(result["end_time"].rstrip("Z")), pd.Timestamp(at(251 - 14).replace(tzinfo=None)))
        # The spike lies in the original values.
        self.assertEqual(max(c[1] for c in curves) > 200, True)

    def test_hit_more_than_30_minutes_after_the_last_is_not_reported_but_counts_as_last(self):
        op, st = self.operator_with(stub_curve())
        st.last_hit = at(250) - datetime.timedelta(minutes=31)
        self.assertIsNone(self.infer(op, 100.0, at(250)))
        self.assertEqual(st.last_hit, at(250))

    def test_hit_exactly_30_minutes_after_the_last_is_reported(self):
        op, st = self.operator_with(stub_curve())
        st.last_hit = at(250) - datetime.timedelta(minutes=30)
        self.assertEqual(len(self.infer(op, 100.0, at(250))), 1)

    def test_after_a_report_the_device_is_locked_for_a_window(self):
        op, st = self.operator_with(stub_curve())
        self.infer(op, 100.0, at(250))
        self.assertEqual(len(self.infer(op, 100.0, at(251))), 1)
        self.assertEqual(st.curve_locked_until, at(251) + CURVE_LOCKOUT)
        self.assertEqual(CURVE_LOCKOUT, datetime.timedelta(minutes=WINDOW))
        # The spike is still in the window: neither a report nor a hit that could feed the next report.
        self.assertIsNone(self.infer(op, 100.0, at(252)))
        self.assertEqual(st.last_hit, at(251))

    def test_the_lock_ends_at_its_time(self):
        op, st = self.operator_with(stub_curve())
        self.infer(op, 100.0, at(250))
        self.infer(op, 100.0, at(251))
        st.curve_locked_until = at(253)
        self.assertIsNone(self.infer(op, 100.0, at(252)))
        self.assertEqual(len(self.infer(op, 100.0, at(253))), 1)  # the hit of 251 is within the debounce

    def test_flat_window_gives_no_hit(self):
        op, st = self.operator_with(stub_curve(), spike=False)
        self.assertIsNone(self.infer(op, 100.0, at(250)))
        self.assertIsNone(st.last_hit)

    def test_no_check_before_a_full_window_is_buffered(self):
        op = self.operator({KEY: profile(curve=stub_curve())}, check_data_anomalies=True,
                           check_receive_time_outlier=False, check_data_extreme_outlier=False)
        for i in range(100):
            self.assertIsNone(self.infer(op, 300.0 if i % 2 else 100.0, at(i)))
        self.assertIsNone(op._live[KEY].last_hit)

    def test_one_check_per_minute(self):
        op, _ = self.operator_with(stub_curve())
        with mock.patch.object(StubCurve, "reconstruct", autospec=True, side_effect=StubCurve.reconstruct) as rec:
            self.infer(op, 100.0, at(250))
            self.infer(op, 100.0, at(250.5))
            self.assertEqual(rec.call_count, 1)
            self.infer(op, 100.0, at(251))
            self.assertEqual(rec.call_count, 2)

    def test_messages_alone_fill_the_buffer_and_the_real_network_runs(self):
        op = self.operator({KEY: profile(curve=self.real_curve)}, check_data_anomalies=True,
                           check_receive_time_outlier=False, check_data_extreme_outlier=False)
        first_hit, reported = None, []
        for i in range(260):
            results = self.infer(op, 100 + 50 * np.sin(2 * np.pi * i / 1440), at(i))
            if first_hit is None and op._live[KEY].last_hit is not None:
                first_hit = i
                self.assertIsNone(results)
            if results:
                reported.append(i)
                self.assertEqual(len(json.loads(results[0]["original_reconstructed_curves"])), WINDOW)
        # Threshold 0: every checked minute is a hit, and the second one is the first report.
        self.assertIsNotNone(first_hit)
        self.assertEqual(reported[0], first_hit + 1)
        self.assertEqual(len(op._live[KEY].buffer), 241)  # 240 minutes, both ends included

    def test_exception_in_the_curve_check_keeps_the_other_findings(self):
        op = self.operator({KEY: profile(curve=stub_curve(RaisingCurve))}, check_data_anomalies=True)
        st = op._live.setdefault(KEY, main.live.DeviceState())
        for k in range(250):
            st.push(at(k), 100.0)
        self.infer(op, 100.0, at(249.5))
        with mock.patch("main.logger") as log:
            results = self.infer(op, 1000.0, at(250))
        log.exception.assert_called_once()
        self.assertEqual([r["type"] for r in results], ["extreme_value", "time"])


class TestContCurveRegression(OperatorTestCase):
    """
    A small real model on a synthetic daily cycle: a normal day stays quiet, a short spike is reported once. The
    epochs are many because the small training set has few steps per epoch; fewer leave a network whose error
    is too large for the spike to stand out.
    """

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model, _, _ = fit_anomaly_model([FakeDataset(to_frame(day_curve(5, step_s=60, seed=1)))],
                                            [{"filterType": "DeviceId", "filterValue": KEY}], T0, True, epochs=300)
        cls.fresh = day_curve(1, step_s=60, seed=2, start=START + pd.Timedelta(days=5)).to_numpy()

    def run_values(self, values):
        """Minutes of the curve_anomaly reports when the values come in minute by minute."""
        op = self.operator(None, model=self.model, check_data_anomalies=True, check_receive_time_outlier=False,
                           check_data_extreme_outlier=False)
        found = []
        for i, v in enumerate(values):
            self.wall = at(i)
            results = self.infer(op, float(v), at(i))
            found += [i for r in results or [] if r["type"] == "curve_anomaly"]
        return found

    def test_a_fresh_normal_day_gives_at_most_two_reports(self):
        self.assertIsNotNone(self.model.profile(KEY).curve)
        self.assertLessEqual(len(self.run_values(self.fresh)), 2)

    def test_a_five_minute_spike_gives_exactly_one_report_in_the_window_after_it(self):
        values = self.fresh.copy()
        values[700:705] += 600
        found = [i for i in self.run_values(values) if 700 <= i <= 700 + WINDOW]
        self.assertEqual(len(found), 1)


class TestLoadCurve(OperatorTestCase):
    def feed_run(self, op, minutes_on, minutes_off=12):
        values = [5.0] * 3 + [300.0] * minutes_on + [5.0] * minutes_off
        results = [self.infer(op, v, at(i)) for i, v in enumerate(values)]
        return [r for r in results if r]

    def operator_load(self):
        curve = load_curve()
        self.assertEqual(curve.score(np.full(200, 300.0)), "uncontinious_device_length")
        self.assertIsNone(curve.score(np.full(40, 300.0)))
        return self.operator({KEY: profile(None, None, curve, "load_device", 5.0)}, check_data_anomalies=True,
                             check_receive_time_outlier=False, check_data_extreme_outlier=False)

    def test_overlong_run_gives_a_length_anomaly_at_its_end(self):
        found = self.feed_run(self.operator_load(), 200)
        self.assertEqual(len(found), 1)
        (result,) = found[0]
        self.assertEqual((result["type"], result["sub_type"], result["value"], result["device_id"]),
                         ("curve_anomaly", "uncontinious_device_length", "", KEY))
        self.assertTrue(result["start_time"].endswith("Z") and result["end_time"].endswith("Z"))
        self.assertNotIn("original_reconstructed_curves", result)

    def test_normal_run_gives_nothing(self):
        self.assertEqual(self.feed_run(self.operator_load(), 40), [])

    def test_run_still_going_gives_nothing(self):
        self.assertEqual(self.feed_run(self.operator_load(), 200, minutes_off=5), [])

    def test_flag_off_gives_nothing(self):
        op = self.operator_load()
        op.config.check_data_anomalies = False
        self.assertEqual(self.feed_run(op, 200), [])

    def test_the_length_at_runtime_is_the_length_in_training_at_any_cadence(self):
        from algo import preprocessing
        for cadence in (1, 5, 15, 60):
            with self.subTest(cadence=cadence):
                index = pd.date_range(T0.replace(tzinfo=None), periods=20 * 60 // cadence, freq=f"{cadence}min")
                values = np.full(len(index), 5.0)
                values[(index >= index[0] + pd.Timedelta(150, "min")) & (index < index[0] + pd.Timedelta(210, "min"))] = 300.0
                expected = preprocessing.extract_loads(preprocessing.to_minutes(pd.Series(values, index=index)), 5.0)
                self.assertEqual(len(expected), 1)
                lengths = []

                class Capture(LoadCurveModel):
                    def score(self, load):
                        lengths.append(len(load))

                base = load_curve()
                curve = Capture(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)})
                op = self.operator({KEY: profile(None, None, curve, "load_device", 5.0)}, check_data_anomalies=True,
                                   check_receive_time_outlier=False, check_data_extreme_outlier=False)
                for t, v in zip(index, values):
                    self.infer(op, float(v), t.to_pydatetime().replace(tzinfo=UTC))
                self.assertEqual(lengths, [len(expected[0])])


class TestRealCaseBimodal(OperatorTestCase):
    def test_pump_and_heating_element(self):
        rng = np.random.RandomState(4)
        values = []
        for _ in range(60):
            values += [0.0] * 120 + list(40 + rng.normal(0, 2, 30)) + list(rng.uniform(1700, 1950, 20))
        index = START + pd.to_timedelta(np.arange(len(values)), unit="min")
        model, _, _ = fit_anomaly_model([FakeDataset(to_frame(pd.Series(values, index=index)))],
                                        [{"filterType": "DeviceId", "filterValue": KEY}], T0, False)
        op = self.operator(model=model, check_receive_time_outlier=False)
        self.assertIsNone(self.infer(op, 0.0, at(0)))  # standby
        self.assertIsNone(self.infer(op, 1950.0, at(1)))
        (result,) = self.infer(op, 3000.0, at(2))
        self.assertEqual((result["type"], result["sub_type"]), ("extreme_value", "high"))


class TestRealCaseGaps(OperatorTestCase):
    """Gap rules on synthetic images of the real devices."""

    def trained(self, times):
        frame = pd.DataFrame({"time": times, "value": 100.0})
        model, _, _ = fit_anomaly_model([FakeDataset(frame)], [{"filterType": "DeviceId", "filterValue": KEY}], T0, False)
        return model

    def test_device_sending_every_4_seconds_with_pauses_of_90_seconds(self):
        seconds, t = [], 0
        for i in range(8000):
            seconds.append(t)
            t += 90 if i % 300 == 299 else 4
        model = self.trained(pd.Timestamp("2026-03-01") + pd.to_timedelta(seconds, unit="s"))
        self.assertEqual(model.profile(KEY).gaps.upper(), 5)
        op = self.operator(model=model, check_data_extreme_outlier=False)
        self.wall = at(0)
        self.assertIsNone(self.infer(op, 100.0, at(0)))
        self.wall = at(1.5)  # 90 seconds of silence
        op._check_outages()
        self.assertEqual(self.produced, [])
        self.assertIsNone(self.infer(op, 100.0, at(1.5)))
        self.wall = at(1.5 + 6)
        op._check_outages()
        (report,) = self.produced
        self.assertEqual((report["type"], report["sub_type"]), ("time", "high"))

    def test_regular_sender_missing_one_message_gives_no_alarm(self):
        model = self.trained(pd.Timestamp("2026-03-01") + pd.to_timedelta(np.arange(3000) * 30, unit="s"))
        op = self.operator(model=model, check_data_extreme_outlier=False)
        self.wall = at(0)
        self.infer(op, 100.0, at(0))
        self.wall = at(1)  # one message of 30 seconds missing
        op._check_outages()
        self.assertEqual(self.produced, [])
        self.assertIsNone(self.infer(op, 100.0, at(1)))


class TestOutages(OperatorTestCase):
    def armed(self, **config):
        op = self.operator({KEY: profile(gaps=GapStats(1.0, 0.1, 1000, 1.2))}, **config)  # upper limit 5 minutes
        self.wall = at(0)
        self.infer(op, 100.0, at(0))
        return op

    def reports(self):
        return [p for p in self.produced if p["type"] == "time"]

    def test_outage_gives_exactly_one_report_even_when_checked_again(self):
        op = self.armed()
        self.wall = at(4)
        op._check_outages()
        self.assertEqual(self.reports(), [])
        self.wall = at(6)
        op._check_outages()
        self.wall = at(7)
        op._check_outages()
        self.assertEqual(self.reports(), [{"type": "time", "sub_type": "high", "value": 6.0, "mean": 1.0,
                                           "threshold": 5.0, "device_id": KEY}])

    def test_a_new_message_arms_it_again(self):
        op = self.armed()
        self.wall = at(6)
        op._check_outages()
        self.wall = at(10)
        self.infer(op, 100.0, at(10))
        self.wall = at(10.5)
        op._check_outages()
        self.assertEqual(len(self.reports()), 1)
        self.wall = at(16)
        op._check_outages()
        self.assertEqual(len(self.reports()), 2)

    def test_no_report_while_paused_and_the_pause_ends_with_a_message_without_lag(self):
        op = self.armed()
        stale = anomaly_model({KEY: profile()}, trained_until=T0 - datetime.timedelta(days=1))
        self.assertTrue(op.need_retraining(FakePyFuncModel(stale)))
        self.assertTrue(op._paused)
        self.wall = at(6)
        op._check_outages()
        self.assertEqual(self.reports(), [])
        # A message with a backlog does not release it.
        self.infer(op, 100.0, self.wall - datetime.timedelta(seconds=LAG_LIMIT_S + 1))
        self.assertTrue(op._paused)
        op._check_outages()
        self.assertEqual(self.reports(), [])
        # One without does.
        self.infer(op, 100.0, self.wall - datetime.timedelta(seconds=LAG_LIMIT_S - 1))
        self.assertFalse(op._paused)
        self.wall = at(12)
        op._check_outages()
        self.assertEqual(len(self.reports()), 1)

    def test_no_report_while_catching_up_on_a_backlog(self):
        op = self.armed()
        self.wall = at(60)
        self.infer(op, 100.0, at(30))  # 30 minutes old
        self.wall = at(65)
        op._check_outages()
        self.assertEqual(self.reports(), [])

    def test_flag_off_gives_no_report(self):
        op = self.armed(check_receive_time_outlier=False)
        self.wall = at(30)
        op._check_outages()
        self.assertEqual(self.reports(), [])

    def test_device_without_statistic_is_not_watched(self):
        op = self.operator(None)
        self.infer(op, 100.0, at(0))
        self.wall = at(500)
        op._check_outages()
        self.assertEqual(self.reports(), [])

    def test_new_device_is_watched_through_the_running_statistic(self):
        op = self.operator(None, check_data_extreme_outlier=False)
        for i in range(MIN_GAPS + 1):
            self.wall = at(10 * i)
            self.infer(op, 100.0, at(10 * i))
        last = 10 * MIN_GAPS
        self.wall = at(last + 29)
        op._check_outages()
        self.assertEqual(self.reports(), [])
        self.wall = at(last + 31)
        op._check_outages()
        (report,) = self.reports()
        self.assertEqual((report["sub_type"], report["value"], report["threshold"]), ("high", 31.0, 30.0))

    def test_failing_produce_does_not_raise(self):
        op = self.armed()
        op.produce = mock.Mock(side_effect=RuntimeError("kafka down"))
        self.wall = at(6)
        with mock.patch("main.logger") as log:
            op._check_outages()
        log.exception.assert_called_once()

    def test_watchdog_thread_reports_and_stop_joins_it(self):
        op = self.armed()
        self.wall = at(6)
        with mock.patch.object(main, "WATCHDOG_INTERVAL_S", 0.01), \
                mock.patch.object(main.MLOperator, "init"), mock.patch.object(main.MLOperator, "stop"), \
                mock.patch.object(Operator, "_restore_arrivals"):
            op.init()
            deadline = time.monotonic() + 10
            while not self.reports() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(self.reports()), 1)
            self.assertTrue(op._watchdog.daemon)
            op.stop()
        self.assertFalse(op._watchdog.is_alive())


class TestRestore(OperatorTestCase):
    """A pod that replaces another during an outage reads the last arrivals back from the history."""

    META = [{"filterType": "DeviceId", "filterValue": KEY}]

    def restored(self, last, devices=None, history=None, ray_get=None):
        op = self.operator(devices or {KEY: profile(gaps=GapStats(1.0, 0.1, 1000, 1.2))})  # upper limit 5 minutes
        if history is None:
            history = [FakeDataset(pd.DataFrame({"time": [last.replace(tzinfo=None) - MINUTE, last.replace(tzinfo=None)],
                                                 "value": [100.0, 100.0]}))]
        self.request = mock.Mock(return_value=history)
        self.ray = mock.Mock()
        self.ray.get.side_effect = ray_get or (lambda refs, timeout=None: refs)
        with mock.patch("main.provide_historic_data", self.request), mock.patch("main.ray", self.ray), \
                mock.patch("main._topic_meta", return_value=self.META):
            op._restore_arrivals(self.wall)
        return op

    def reports(self):
        return [p for p in self.produced if p["type"] == "time"]

    def test_an_outage_that_began_before_the_start_is_reported(self):
        op = self.restored(last=at(-2))
        self.wall = at(2.5)
        op._check_outages()
        self.assertEqual(self.reports(), [])
        self.wall = at(3.5)
        op._check_outages()
        self.wall = at(4)
        op._check_outages()
        self.assertEqual(self.reports(), [{"type": "time", "sub_type": "high", "value": 5.5, "mean": 1.0,
                                           "threshold": 5.0, "device_id": KEY}])
        self.ray.init.assert_called_once()
        self.ray.shutdown.assert_called_once()

    def test_a_device_past_its_bound_at_the_start_is_left_to_the_pod_before(self):
        op = self.restored(last=at(-5))
        self.wall = at(30)
        op._check_outages()
        self.assertEqual(self.reports(), [])
        self.assertNotIn(KEY, op._live)

    def test_a_live_arrival_is_kept(self):
        op = self.operator({KEY: profile(gaps=GapStats(1.0, 0.1, 1000, 1.2))})
        self.infer(op, 100.0, at(0))
        frame = pd.DataFrame({"time": [at(-1).replace(tzinfo=None)], "value": [100.0]})
        with mock.patch("main.provide_historic_data", return_value=[FakeDataset(frame)]), \
                mock.patch("main.ray", mock.Mock(get=lambda refs, timeout=None: refs)), \
                mock.patch("main._topic_meta", return_value=self.META):
            op._restore_arrivals(at(0))
        self.assertEqual(op._live[KEY].last_arrival, at(0))

    def test_the_window_is_the_widest_bound_and_a_margin_at_most_restore_max(self):
        devices = {KEY: profile(gaps=GapStats(1.0, 0.1, 1000, 1.2)), "b": profile(gaps=GapStats(10.0, 1.0, 1000, 12.0)),
                   "c": profile(gaps=None)}
        self.restored(last=at(-2), devices=devices)
        self.request.assert_called_once_with(datetime.timedelta(minutes=36) + main.RESTORE_MARGIN)
        self.restored(last=at(-2), devices={KEY: profile(gaps=GapStats(600.0, 1.0, 1000, 6000.0))})
        self.request.assert_called_once_with(main.RESTORE_MAX)

    def test_a_device_the_model_has_no_bound_for_is_not_armed_and_the_others_are(self):
        frame = pd.DataFrame({"time": [at(-1).replace(tzinfo=None)], "value": [100.0]})
        self.META = [{"filterType": "DeviceId", "filterValue": "other"}, {"filterType": "DeviceId", "filterValue": KEY}]
        op = self.restored(last=at(-1), history=[FakeDataset(frame), FakeDataset(frame)])
        self.assertNotIn("other", op._live)
        self.assertEqual(op._live[KEY].last_arrival, at(-1))

    def test_an_unreadable_history_does_not_stop_the_start(self):
        def timeout(refs, timeout=None):
            raise TimeoutError("no worker free")
        with mock.patch("main.logger") as log:
            op = self.restored(last=at(-2), ray_get=timeout)
        log.exception.assert_called_once()
        self.ray.shutdown.assert_called_once()
        self.assertEqual(op._live, {})

    def test_init_restores_from_its_own_start_before_consuming_unless_switched_off(self):
        for flag, calls in ((True, [mock.call(at(0))]), (False, [])):
            op = self.operator({KEY: profile()}, check_receive_time_outlier=flag)
            self.wall = at(0)
            with mock.patch.object(main.MLOperator, "init", side_effect=lambda *a, **k: setattr(self, "wall", at(3))), \
                    mock.patch.object(Operator, "_restore_arrivals") as restore:
                op.init()
            op._stop_event.set()
            self.assertEqual(restore.call_args_list, calls)


class TestSchema(OperatorTestCase):
    def test_schema_error_is_produced_with_the_flag(self):
        op = self.operator(None, check_data_schema=True)
        produce = mock.Mock()
        op.handle_result_error(ValueError("x"), {"a": {"b": 1}}, produce, "d1")
        produce.assert_called_once_with({"type": "schema", "sub_type": "", "value": json.dumps({"a": {"b": 1}}),
                                         "device_id": "d1"})

    def test_schema_error_is_dropped_without_the_flag(self):
        op = self.operator(None, check_data_schema=False)
        produce = mock.Mock()
        op.handle_result_error(ValueError("x"), {"a": 1}, produce, "d1")
        produce.assert_not_called()


class TestConfig(unittest.TestCase):
    NAMES = ("check_data_anomalies", "check_data_extreme_outlier", "check_data_schema", "check_receive_time_outlier")
    DEFAULTS = [False, True, True, True]

    def values(self, config):
        return [getattr(config, n) for n in self.NAMES]

    def test_defaults_without_values(self):
        c = CustomConfig({})
        self.assertEqual(self.values(c), self.DEFAULTS)
        self.assertEqual(c.logger_level, "warning")

    def test_empty_string_and_none_give_the_default(self):
        self.assertEqual(self.values(CustomConfig({n: "" for n in self.NAMES})), self.DEFAULTS)
        self.assertEqual(self.values(CustomConfig({n: None for n in self.NAMES})), self.DEFAULTS)

    def test_strings(self):
        for raw, expected in (("True", True), ("true", True), ("TRUE", True), (" 1 ", True), ("false", False),
                              ("False", False), ("0", False), ("no", False)):
            with self.subTest(raw=raw):
                self.assertEqual(self.values(CustomConfig({n: raw for n in self.NAMES})), [expected] * 4)

    def test_real_bools_pass_through(self):
        c = CustomConfig({"check_data_anomalies": True, "check_data_extreme_outlier": False})
        self.assertIs(c.check_data_anomalies, True)
        self.assertIs(c.check_data_extreme_outlier, False)


def history_frame(minutes=300, seed=0):
    rng = np.random.RandomState(seed)
    t = pd.Timestamp("2026-03-01") + pd.to_timedelta(np.arange(minutes), unit="m")
    return pd.DataFrame({"time": t, "value": 100 + rng.normal(0, 2, minutes)})


class TestTrain(OperatorTestCase):
    META = [{"filterType": "DeviceId", "filterValue": "d1"}]

    def run_train(self, op, model=None, history=None, remote=None, ray_get=None, history_error=None, logger=None):
        remote = remote or (lambda *a: fit_anomaly_model(*a))
        request = (mock.patch("main.provide_historic_data", side_effect=history_error) if history_error
                   else mock.patch("main.provide_historic_data", return_value=history))
        with request, mock.patch("main._topic_meta", return_value=self.META), \
                mock.patch("main.train_anomaly_model", mock.Mock(remote=remote)), \
                mock.patch("main.ray.get", side_effect=ray_get or (lambda r: r)):
            return op.train(model, logger)

    def test_good_history_gives_a_model_and_the_task_gets_the_flag_and_the_meta(self):
        op = self.operator(None, check_data_anomalies=True)
        remote = mock.Mock(side_effect=lambda *a: fit_anomaly_model(*a))
        model = self.run_train(op, history=[FakeDataset(history_frame())], remote=remote)
        self.assertEqual(list(model.devices), ["d1"])
        self.assertEqual(model.trained_until, T0)
        args = remote.call_args.args
        self.assertEqual((args[1], args[2], args[3]), (self.META, T0, True))
        self.assertIsNone(op._retry_at)

    def test_the_three_counters_are_logged(self):
        devices = {"a": profile(curve=stub_curve()), "b": profile(curve=load_curve(), device_type="load_device"),
                   "c": profile()}
        op = self.operator(None)
        logger = mock.Mock()
        model = self.run_train(op, history=[object()], remote=lambda *a: (anomaly_model(devices), ["note"], False),
                               logger=logger)
        self.assertEqual(len(model.devices), 3)
        logger.log_metrics.assert_called_once_with(
            {"devices_trained": 3, "devices_with_curve_model": 1, "devices_with_load_model": 1})

    def test_a_failing_metrics_store_does_not_stop_the_training(self):
        op = self.operator(None)
        logger = mock.Mock()
        logger.log_metrics.side_effect = RuntimeError("mlflow down")
        with mock.patch("main.logger") as log:
            model = self.run_train(op, history=[FakeDataset(history_frame())], logger=logger)
        self.assertEqual(list(model.devices), ["d1"])
        log.exception.assert_called_once()

    def test_the_outage_report_resumes_after_the_grace_period_without_a_message(self):
        op = self.operator({KEY: profile(gaps=GapStats(1.0, 0.1, 1000, 1.2))})  # upper limit 5 minutes
        self.wall = at(0)
        self.infer(op, 100.0, at(0))
        stale = anomaly_model({KEY: profile()}, trained_until=T0 - datetime.timedelta(days=1))
        self.assertTrue(op.need_retraining(FakePyFuncModel(stale)))
        self.assertIsNone(op._pause_until)
        now = [1000.0]
        with mock.patch.object(main, "time", mock.Mock(monotonic=lambda: now[0])):
            self.assertIsNotNone(self.run_train(op, model=FakePyFuncModel(stale), history=[FakeDataset(history_frame())]))
            self.assertEqual(op._pause_until, 1000.0 + main.PAUSE_GRACE_S)
            self.wall = at(30)
            op._check_outages()
            self.assertEqual(self.produced, [])  # still within the grace period
            now[0] += main.PAUSE_GRACE_S - 1
            op._check_outages()
            self.assertEqual(self.produced, [])
            now[0] += 2
            op._check_outages()
        self.assertEqual([p["sub_type"] for p in self.produced], ["high"])

    def test_every_way_out_of_the_training_starts_the_grace_period(self):
        op = self.operator(None)
        self.assertIsNone(self.run_train(op, history=[UnreadableDataset()]))  # retry path
        self.assertIsNotNone(op._pause_until)
        op = self.operator(None)
        self.assertIsNotNone(self.run_train(op, history=[FakeDataset(history_frame())]))
        self.assertIsNotNone(op._pause_until)

    def test_no_logger_is_fine(self):
        op = self.operator(None)
        self.assertIsNotNone(self.run_train(op, history=[FakeDataset(history_frame())], logger=None))

    def test_failing_task_gives_none_and_a_retry_time(self):
        op = self.operator(None)
        with mock.patch("main.logger") as log:
            model = self.run_train(op, history=[FakeDataset(history_frame())], ray_get=RuntimeError("no CPU"))
        log.exception.assert_called_once()
        self.assertIsNone(model)
        self.assertEqual(op._history_retries, 1)
        self.assertGreater(op._retry_at, time.monotonic() + HISTORY_RETRY_S - 5)

    def test_unreadable_history_gives_none(self):
        op = self.operator(None)
        self.assertIsNone(self.run_train(op, history=[UnreadableDataset()]))
        self.assertEqual(op._history_retries, 1)

    def test_failing_history_request_gives_none(self):
        op = self.operator(None)
        self.assertIsNone(self.run_train(op, history_error=RuntimeError("no CONFIG")))
        self.assertEqual(op._history_retries, 1)

    def test_after_the_retries_a_task_failure_gives_an_empty_model(self):
        op = self.operator(None)
        with mock.patch("main.logger"):
            for _ in range(HISTORY_RETRIES):
                self.assertIsNone(self.run_train(op, history=[object()], ray_get=RuntimeError("x")))
            model = self.run_train(op, history=[object()], ray_get=RuntimeError("x"))
        self.assertEqual((model.devices, model.trained_until, model.schema_version), ({}, T0, SCHEMA_VERSION))
        self.assertEqual((op._history_retries, op._retry_at), (0, None))
        self.assertFalse(op.need_retraining(FakePyFuncModel(model)))

    def test_after_the_retries_the_previous_devices_stay(self):
        previous = anomaly_model({"old": profile(), "d1": profile(values=None)},
                                 trained_until=T0 - datetime.timedelta(days=1))
        op = self.operator(None, model=previous)
        for _ in range(HISTORY_RETRIES):
            self.run_train(op, model=FakePyFuncModel(previous), history=[UnreadableDataset()])
        model = self.run_train(op, model=FakePyFuncModel(previous), history=[UnreadableDataset()])
        self.assertEqual(model.trained_until, T0)
        self.assertEqual(set(model.devices), {"old", "d1"})

    def test_a_readable_but_thin_history_replaces_the_model(self):
        previous = anomaly_model({"old": profile()}, trained_until=T0 - datetime.timedelta(days=1))
        op = self.operator(None, model=previous)
        model = self.run_train(op, model=FakePyFuncModel(previous), history=[FakeDataset(history_frame(5))])
        self.assertEqual(list(model.devices), ["d1"])
        self.assertEqual(op._history_retries, 0)

    def test_success_resets_the_retry_state(self):
        op = self.operator(None)
        self.run_train(op, history=[UnreadableDataset()])
        self.assertIsNotNone(op._retry_at)
        self.assertIsNotNone(self.run_train(op, history=[FakeDataset(history_frame())]))
        self.assertEqual((op._history_retries, op._retry_at), (0, None))

    def test_topic_meta_comes_from_the_operator_config(self):
        cfg = {"config": {}, "inputTopics": [{"name": "t", "filterType": "DeviceId", "filterValue": "d1,d2",
                                              "mappings": [{"dest": "value", "source": "x"}]}]}
        with mock.patch("main.util.load_operator_config_json", return_value=cfg), \
                mock.patch("main.util.DeploymentConfig"):
            self.assertEqual(main._topic_meta(), [{"filterType": "DeviceId", "filterValue": "d1,d2"}])


class TestNeedRetraining(OperatorTestCase):
    def check(self, op, model):
        return op.need_retraining(FakePyFuncModel(model) if model is not None else None)

    def test_model_of_today_in_berlin_needs_none(self):
        op = self.operator(None)
        self.assertFalse(self.check(op, anomaly_model({}, trained_until=T0 - datetime.timedelta(hours=3))))
        self.assertFalse(op._paused)

    def test_the_day_is_the_berlin_day(self):
        # 00:30 in Berlin on the 11th and 10:00 on the 11th: one day, though UTC says the 10th and the 11th.
        trained = datetime.datetime(2026, 3, 10, 23, 30, tzinfo=UTC)
        clock.set_fixed(datetime.datetime(2026, 3, 11, 10, 0, tzinfo=UTC))
        self.assertFalse(self.check(self.operator(None), anomaly_model({}, trained_until=trained)))
        # 22:30 UTC on the 10th is the 10th in UTC and the 10th in Berlin; 23:30 is already the 11th in Berlin.
        clock.set_fixed(datetime.datetime(2026, 3, 10, 22, 30, tzinfo=UTC))
        self.assertTrue(self.check(self.operator(None), anomaly_model({}, trained_until=trained)))

    def test_model_of_yesterday_needs_training_and_pauses(self):
        op = self.operator(None)
        self.assertTrue(self.check(op, anomaly_model({}, trained_until=T0 - datetime.timedelta(days=1))))
        self.assertTrue(op._paused)

    def test_foreign_schema_version_needs_training(self):
        op = self.operator(None)
        self.assertTrue(self.check(op, anomaly_model({}, schema_version=SCHEMA_VERSION + 1)))

    def test_no_model_needs_training(self):
        self.assertTrue(self.check(self.operator(None), None))

    def test_open_retry_in_the_future_waits(self):
        op = self.operator(None)
        op._retry_at = time.monotonic() + 100
        self.assertFalse(self.check(op, anomaly_model({}, trained_until=T0 - datetime.timedelta(days=1))))
        self.assertFalse(self.check(op, None))
        self.assertFalse(op._paused)

    def test_open_retry_in_the_past_is_due_even_with_a_model_of_today(self):
        op = self.operator(None)
        op._retry_at = time.monotonic() - 1
        self.assertTrue(self.check(op, anomaly_model({})))
        self.assertTrue(op._paused)


class TestManifest(unittest.TestCase):
    def test_manifest_has_exactly_the_ports_and_config_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "manifest.json")
            done = subprocess.run([sys.executable, "-m", "operator_lib", "manifest", "main.py", "-o", out],
                                  cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(done.returncode, 0, done.stderr)
            with open(out) as f:
                manifest = json.load(f)

        def entries(*pairs):
            return [{"name": n, "type": t} for n, t in pairs]

        self.assertEqual(manifest, {
            "inputs": entries(("value", "float"), ("original_input_ids", "string")),
            "outputs": entries(("type", "string"), ("sub_type", "string"), ("value", "string"),
                               ("initial_phase", "string"), ("mean", "float"), ("threshold", "float"),
                               ("device_id", "string"), ("start_time", "string"), ("end_time", "string"),
                               ("original_reconstructed_curves", "string")),
            "config_values": entries(("logger_level", "string"), ("check_data_anomalies", "bool"),
                                     ("check_data_extreme_outlier", "bool"), ("check_data_schema", "bool"),
                                     ("check_receive_time_outlier", "bool")),
        })


if __name__ == "__main__":
    unittest.main()
