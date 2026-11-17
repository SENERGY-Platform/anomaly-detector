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
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from operator_lib.util import clock
from operator_lib.util.helpers import HistoryNeedsRayError

import main
from main import HISTORY_RETRIES, HISTORY_RETRY_S, LAG_LIMIT_S, CustomConfig, Operator
from algo.history import last_arrivals
from algo.stats import MIN_SAMPLES, GapStats, ValueStats, value_stats
from algo.model import SCHEMA_VERSION, AnomalyModel, DeviceProfile
from algo.training import fit_anomaly_model
from tests.test_training import START, to_frame

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


def profile(values=ValueStats(100.0, 65.0, 135.0), gaps=GapStats(10.0, 12.0)):
    return DeviceProfile(values, gaps)


def anomaly_model(devices, trained_until=T0, schema_version=SCHEMA_VERSION):
    return AnomalyModel(schema_version, trained_until, devices)


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

    def test_a_short_gap_is_not_reported(self):
        op = self.operator({KEY: profile()})
        self.assertIsNone(self.infer(op, 100.0, at(0)))
        self.assertIsNone(self.infer(op, 100.0, at(0.01)))

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

    def test_device_in_standby_reports_no_low(self):
        # Standby at the 0.01 % quantile puts the lower bound below zero.
        op = self.operator({KEY: profile(ValueStats(300.0, 5.0, 800.0), None)})
        self.assertIsNone(self.infer(op, 5.0, T0))
        self.assertIsNone(self.infer(op, 0.0, at(1)))

    def test_new_device_reports_through_the_running_statistic_after_min_values(self):
        op = self.operator(None, check_receive_time_outlier=False)
        values = [99.0 + 2 * (i % 2) for i in range(MIN_SAMPLES)]
        for i, v in enumerate(values):
            self.assertIsNone(self.infer(op, v, at(i)))
        (result,) = self.infer(op, 120.0, at(MIN_SAMPLES))
        expected = value_stats(np.array(values))  # the calculation of the training
        self.assertEqual((result["sub_type"], result["threshold"], result["mean"]),
                         ("high", round(expected.upper(), 2), 100.0))
        self.assertAlmostEqual(result["threshold"], 106.05, places=2)  # floor: 5 % of 101

    def test_new_device_reports_a_new_normal_level_once(self):
        op = self.operator(None, check_receive_time_outlier=False)
        for i in range(MIN_SAMPLES):
            self.infer(op, 200.0 + (i % 3 - 1), at(i))
        reports = [self.infer(op, 230.0, at(MIN_SAMPLES + i)) for i in range(20)]
        self.assertEqual(sum(r is not None for r in reports), 1)

    def test_new_device_reports_nothing_before_min_values(self):
        op = self.operator(None, check_receive_time_outlier=False)
        for i in range(MIN_SAMPLES - 1):
            self.infer(op, 100.0, at(i))
        self.assertIsNone(self.infer(op, 5000.0, at(MIN_SAMPLES)))

    def test_the_profile_is_never_changed_by_a_message(self):
        p = profile()
        op = self.operator({KEY: p})
        self.infer(op, 180.0, T0)
        self.assertEqual(p.values, ValueStats(100.0, 65.0, 135.0))


class TestGap(OperatorTestCase):
    def test_out_of_order_message_is_no_gap(self):
        op = self.operator({KEY: profile()})
        self.infer(op, 100.0, at(10))
        self.assertIsNone(self.infer(op, 100.0, at(5)))
        self.assertEqual(op._devices[KEY].last_arrival, at(10))


class TestRealCaseBimodal(OperatorTestCase):
    def test_pump_and_heating_element(self):
        rng = np.random.RandomState(4)
        values = []
        for _ in range(60):
            values += [0.0] * 120 + list(40 + rng.normal(0, 2, 30)) + list(rng.uniform(1700, 1950, 20))
        index = START + pd.to_timedelta(np.arange(len(values)), unit="min")
        model, _, _ = fit_anomaly_model([FakeDataset(to_frame(pd.Series(values, index=index)))],
                                        [{"filterType": "DeviceId", "filterValue": KEY}], T0)
        op = self.operator(model=model, check_receive_time_outlier=False)
        self.assertIsNone(self.infer(op, 0.0, at(0)))  # standby
        self.assertIsNone(self.infer(op, 1950.0, at(1)))
        (result,) = self.infer(op, 3000.0, at(2))
        self.assertEqual((result["type"], result["sub_type"]), ("extreme_value", "high"))


class TestRealCaseGaps(OperatorTestCase):
    """Gap rules on synthetic images of the real devices."""

    def trained(self, times):
        frame = pd.DataFrame({"time": times, "value": 100.0})
        model, _, _ = fit_anomaly_model([FakeDataset(frame)], [{"filterType": "DeviceId", "filterValue": KEY}], T0)
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
        op = self.operator({KEY: profile(gaps=GapStats(1.0, 1.2))}, **config)  # upper limit 5 minutes
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

    def test_a_late_message_of_one_device_does_not_hide_the_outage_of_another(self):
        silent_after_5 = profile(gaps=GapStats(1.0, 1.2))
        op = self.operator({"a": silent_after_5, "b": silent_after_5})
        self.wall = at(0)
        self.infer(op, 100.0, at(0), device_id="a")
        self.infer(op, 100.0, at(-30), device_id="b")  # sent 30 minutes ago
        self.wall = at(6)
        op._check_outages()
        self.assertIn("a", [r["device_id"] for r in self.reports()])

    def test_a_message_out_of_order_still_shows_the_device_is_alive(self):
        op = self.armed()
        self.wall = at(10)
        self.infer(op, 100.0, at(10))
        self.wall = at(10.5)
        self.infer(op, 100.0, at(5))
        self.wall = at(15)
        op._check_outages()
        self.assertEqual(self.reports(), [])
        self.wall = at(16)
        op._check_outages()
        self.assertEqual([r["value"] for r in self.reports()], [5.5])

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
        for i in range(MIN_SAMPLES + 1):
            self.wall = at(10 * i)
            self.infer(op, 100.0, at(10 * i))
        last = 10 * MIN_SAMPLES
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

    def restored(self, last, devices=None, history=None, ray_get=None, frames_error=None, datasets=None):
        """frames_error: what reading the history in the pod raises, after which it is read in a Ray task;
        datasets: what the Ray task reads, the history by default."""
        op = self.operator(devices or {KEY: profile(gaps=GapStats(1.0, 1.2))})  # upper limit 5 minutes
        if history is None:
            history = [pd.DataFrame({"time": [last.replace(tzinfo=None) - MINUTE, last.replace(tzinfo=None)],
                                     "value": [100.0, 100.0]})]
        self.frames = mock.Mock(side_effect=frames_error) if frames_error else mock.Mock(return_value=history)
        self.request = mock.Mock(return_value=datasets or [FakeDataset(frame) for frame in history])
        self.ray = mock.Mock()
        self.ray.get.side_effect = ray_get or (lambda ref, timeout=None: ref)
        self.task = mock.Mock(remote=mock.Mock(side_effect=last_arrivals))
        with mock.patch("main.provide_historic_frames_local", self.frames), \
                mock.patch("main.provide_historic_data", self.request), mock.patch("main.ray", self.ray), \
                mock.patch("main.read_last_arrivals", self.task), mock.patch("main._topic_meta", return_value=self.META):
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

    def test_the_history_is_read_in_the_pod_without_ray(self):
        self.restored(last=at(-2))
        self.frames.assert_called_once()
        self.request.assert_not_called()
        self.ray.init.assert_not_called()
        self.task.remote.assert_not_called()

    def test_an_input_only_kafka_has_is_read_in_a_ray_task(self):
        op = self.restored(last=at(-2), frames_error=HistoryNeedsRayError("only in Kafka"))
        self.task.remote.assert_called_once_with(self.request.return_value, self.META)
        self.assertEqual(self.ray.get.call_args.kwargs, {"timeout": main.RESTORE_TIMEOUT_S})
        self.ray.init.assert_called_once()
        self.ray.shutdown.assert_called_once()
        self.assertEqual(op._devices[KEY].last_arrival, at(-2))
        self.assertTrue(op._devices[KEY].armed)

    def test_a_device_past_its_bound_at_the_start_is_left_to_the_pod_before(self):
        op = self.restored(last=at(-5))
        self.wall = at(30)
        op._check_outages()
        self.assertEqual(self.reports(), [])
        self.assertNotIn(KEY, op._devices)

    def test_a_live_arrival_is_kept(self):
        op = self.operator({KEY: profile(gaps=GapStats(1.0, 1.2))})
        self.infer(op, 100.0, at(0))
        frame = pd.DataFrame({"time": [at(-1).replace(tzinfo=None)], "value": [100.0]})
        with mock.patch("main.provide_historic_frames_local", return_value=[frame]), \
                mock.patch("main._topic_meta", return_value=self.META):
            op._restore_arrivals(at(0))
        self.assertEqual(op._devices[KEY].last_arrival, at(0))

    def test_the_window_is_the_widest_bound_and_a_margin_at_most_restore_max(self):
        devices = {KEY: profile(gaps=GapStats(1.0, 1.2)), "b": profile(gaps=GapStats(10.0, 12.0)),
                   "c": profile(gaps=None)}
        self.restored(last=at(-2), devices=devices)
        self.frames.assert_called_once_with(datetime.timedelta(minutes=36) + main.RESTORE_MARGIN)
        self.restored(last=at(-2), devices={KEY: profile(gaps=GapStats(600.0, 6000.0))})
        self.frames.assert_called_once_with(main.RESTORE_MAX)
        self.restored(last=at(-2), frames_error=HistoryNeedsRayError("only in Kafka"))
        self.request.assert_called_once_with(datetime.timedelta(minutes=5) + main.RESTORE_MARGIN)

    def test_a_device_the_model_has_no_bound_for_is_not_armed_and_the_others_are(self):
        frame = pd.DataFrame({"time": [at(-1).replace(tzinfo=None)], "value": [100.0]})
        self.META = [{"filterType": "DeviceId", "filterValue": "other"}, {"filterType": "DeviceId", "filterValue": KEY}]
        op = self.restored(last=at(-1), history=[frame, frame])
        self.assertNotIn("other", op._devices)
        self.assertEqual(op._devices[KEY].last_arrival, at(-1))

    def test_an_input_unreadable_in_the_pod_leaves_out_only_that_input_in_the_ray_task(self):
        frame = pd.DataFrame({"time": [at(-1).replace(tzinfo=None)], "value": [100.0]})
        self.META = [{"filterType": "DeviceId", "filterValue": "other"}, {"filterType": "DeviceId", "filterValue": KEY}]
        devices = {KEY: profile(gaps=GapStats(1.0, 1.2)), "other": profile(gaps=GapStats(1.0, 1.2))}
        with mock.patch("main.logger") as log:
            op = self.restored(last=at(-1), devices=devices, frames_error=RuntimeError('relation "x" does not exist'),
                               datasets=[UnreadableDataset(), FakeDataset(frame)])
        log.exception.assert_not_called()
        log.warning.assert_called_once()
        self.task.remote.assert_called_once()
        self.assertNotIn("other", op._devices)
        self.assertEqual(op._devices[KEY].last_arrival, at(-1))

    def test_a_read_in_the_pod_that_runs_out_ends_the_restore_without_ray(self):
        release = threading.Event()

        def hanging(duration):
            release.wait(5)
            return []
        with mock.patch.object(main, "RESTORE_TIMEOUT_S", 0.05), mock.patch("main.logger") as log:
            op = self.restored(last=at(-2), frames_error=hanging)
        release.set()
        log.exception.assert_called_once()
        self.ray.init.assert_not_called()
        self.assertEqual(op._devices, {})

    def test_a_ray_task_that_does_not_finish_does_not_stop_the_start(self):
        def timeout(refs, timeout=None):
            raise TimeoutError("no worker free")
        with mock.patch("main.logger") as log:
            op = self.restored(last=at(-2), ray_get=timeout, frames_error=HistoryNeedsRayError("only in Kafka"))
        log.exception.assert_called_once()
        self.ray.cancel.assert_called_once()
        self.ray.shutdown.assert_called_once()
        self.assertEqual(op._devices, {})

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
    NAMES = ("check_data_extreme_outlier", "check_data_schema", "check_receive_time_outlier")
    DEFAULTS = [True, True, True]

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
                self.assertEqual(self.values(CustomConfig({n: raw for n in self.NAMES})), [expected] * 3)

    def test_real_bools_pass_through(self):
        c = CustomConfig({"check_data_schema": True, "check_data_extreme_outlier": False})
        self.assertIs(c.check_data_schema, True)
        self.assertIs(c.check_data_extreme_outlier, False)


def history_frame(minutes=300, seed=0):
    rng = np.random.RandomState(seed)
    t = pd.Timestamp("2026-03-01") + pd.to_timedelta(np.arange(minutes), unit="m")
    return pd.DataFrame({"time": t, "value": 100 + rng.normal(0, 2, minutes)})


class TestTrain(OperatorTestCase):
    META = [{"filterType": "DeviceId", "filterValue": "d1"}]

    def run_train(self, op, model=None, history=None, remote=None, ray_get=None, history_error=None, logger=None,
                  frames=None, frames_error=HistoryNeedsRayError("only in Kafka")):
        """frames: the history read in the pod; without it, reading in the pod raises frames_error and the history
        (datasets) is read in a Ray task."""
        remote = remote or (lambda *a: fit_anomaly_model(*a))
        request = (mock.patch("main.provide_historic_data", side_effect=history_error) if history_error
                   else mock.patch("main.provide_historic_data", return_value=history))
        in_pod = (mock.patch("main.provide_historic_frames_local", return_value=frames) if frames is not None
                  else mock.patch("main.provide_historic_frames_local", side_effect=frames_error))
        with request, in_pod, mock.patch("main._topic_meta", return_value=self.META), \
                mock.patch("main.train_anomaly_model", mock.Mock(remote=remote)), \
                mock.patch("main.ray.get", side_effect=ray_get or (lambda r: r)):
            return op.train(model, logger)

    def test_the_history_is_read_and_learned_in_the_pod_without_ray(self):
        op = self.operator(None)
        remote = mock.Mock()
        model = self.run_train(op, frames=[history_frame()], remote=remote)
        self.assertEqual(list(model.devices), ["d1"])
        self.assertIsNotNone(model.profile("d1").values)
        remote.assert_not_called()

    def test_an_input_only_kafka_has_is_learned_in_a_ray_task(self):
        op = self.operator(None)
        remote = mock.Mock(side_effect=lambda *a: fit_anomaly_model(*a))
        model = self.run_train(op, history=[FakeDataset(history_frame())], remote=remote,
                               frames_error=HistoryNeedsRayError("only in Kafka"))
        self.assertEqual(list(model.devices), ["d1"])
        remote.assert_called_once()

    def test_a_read_in_the_pod_that_runs_out_is_tried_again_later_without_ray(self):
        op = self.operator(None)
        remote = mock.Mock()
        with mock.patch.object(main, "HISTORY_TIMEOUT_S", 0.05), mock.patch("main.logger"):
            model = self.run_train(op, remote=remote, frames_error=lambda *a: time.sleep(1))
        self.assertIsNone(model)
        self.assertEqual(op._history_retries, 1)
        remote.assert_not_called()

    def test_the_bounds_of_every_device_are_logged_as_a_table(self):
        op = self.operator(None)
        logger = mock.Mock()
        self.run_train(op, frames=[history_frame()], logger=logger)
        table, artifact = logger.log_table.call_args.args
        self.assertEqual(artifact, "profiles.json")
        self.assertEqual(list(table["device_id"]), ["d1"])
        self.assertLess(table["value_lower"][0], 100)
        self.assertGreater(table["value_upper"][0], 100)
        self.assertEqual(table["gap_upper_min"][0], 5)

    def test_good_history_gives_a_model_and_the_task_gets_the_meta(self):
        op = self.operator(None)
        remote = mock.Mock(side_effect=lambda *a: fit_anomaly_model(*a))
        model = self.run_train(op, history=[FakeDataset(history_frame())], remote=remote)
        self.assertEqual(list(model.devices), ["d1"])
        self.assertEqual(model.trained_until, T0)
        args = remote.call_args.args
        self.assertEqual((args[1], args[2]), (self.META, T0))
        self.assertIsNone(op._retry_at)

    def test_the_device_count_is_logged(self):
        devices = {"a": profile(), "b": profile(), "c": profile()}
        op = self.operator(None)
        logger = mock.Mock()
        model = self.run_train(op, history=[object()], remote=lambda *a: (anomaly_model(devices), ["note"], False),
                               logger=logger)
        self.assertEqual(len(model.devices), 3)
        logger.log_metrics.assert_called_once_with({"devices_trained": 3})

    def test_a_failing_metrics_store_does_not_stop_the_training(self):
        op = self.operator(None)
        logger = mock.Mock()
        logger.log_metrics.side_effect = RuntimeError("mlflow down")
        with mock.patch("main.logger") as log:
            model = self.run_train(op, history=[FakeDataset(history_frame())], logger=logger)
        self.assertEqual(list(model.devices), ["d1"])
        log.exception.assert_called_once()

    def test_the_outage_report_resumes_after_the_grace_period_without_a_message(self):
        op = self.operator({KEY: profile(gaps=GapStats(1.0, 1.2))})  # upper limit 5 minutes
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
                               ("mean", "float"), ("threshold", "float"), ("device_id", "string")),
            "config_values": entries(("logger_level", "string"), ("check_data_extreme_outlier", "bool"),
                                     ("check_data_schema", "bool"), ("check_receive_time_outlier", "bool")),
        })


if __name__ == "__main__":
    unittest.main()
