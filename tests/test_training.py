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

import cloudpickle
import numpy as np
import pandas as pd

from algo import preprocessing
from algo.detectors import continuous_curve, extreme_value, receive_time
from algo.detectors.continuous_curve import ContCurveModel
from algo.detectors.extreme_value import ValueStats
from algo.detectors.load_curve import CURVE_MIN_TOLERANCE, LoadCurveModel, fit_load_curve
from algo.detectors.receive_time import GapStats
from algo.model import SCHEMA_VERSION, WINDOW, AnomalyModel
from algo.training import fit_anomaly_model

NOW = datetime.datetime(2026, 1, 10, tzinfo=datetime.timezone.utc)
START = pd.Timestamp("2026-01-01")
EPOCHS = 30


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


def load_device_series(n_runs=20, seed=0, long_run=10):
    """Standby of about 2 W, one value per minute, runs of 30 to 40 minutes at about 800 W every 4 hours,
    the run number long_run three times as long."""
    rng = np.random.RandomState(seed)
    values = []
    for i in range(n_runs):
        length = int(rng.randint(30, 41)) * (3 if i == long_run else 1)
        values += list(2 + rng.normal(0, 0.3, 240 - 40 + 5))
        values += list(800 + rng.normal(0, 10, length))
        values += list(2 + rng.normal(0, 0.3, 40))
    index = START + pd.to_timedelta(np.arange(len(values)), unit="min")
    return pd.Series(values, index=index)


def fit(datasets, meta=None, curves=True, epochs=EPOCHS):
    meta = meta if meta is not None else [{"filterType": "DeviceId", "filterValue": "d1"}] * len(datasets or [])
    return fit_anomaly_model(datasets, meta, NOW, curves, epochs=epochs)


class ContinuousDevice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = day_curve(5)
        cls.train_part = cls.series[:START + pd.Timedelta(4, "D")]
        cls.model, cls.notes, cls.failed = fit([FakeDataset(to_frame(cls.train_part))])

    def test_profile(self):
        self.assertFalse(self.failed)
        self.assertEqual(self.model.schema_version, SCHEMA_VERSION)
        self.assertEqual(self.model.trained_until, NOW)
        self.assertEqual(self.model.predict(None, None), ["d1"])
        p = self.model.profile("d1")
        self.assertEqual(p.device_type, "cont_device")
        self.assertIsInstance(p.values, ValueStats)
        self.assertIsInstance(p.gaps, GapStats)
        self.assertAlmostEqual(p.gaps.mean_min, 0.5, places=3)
        self.assertAlmostEqual(p.values.mean, 100, delta=3)
        self.assertGreater(p.values.upper(), p.values.mean)
        self.assertLess(p.values.lower(), p.values.mean)
        self.assertIsInstance(p.curve, ContCurveModel)
        self.assertTrue(np.isfinite(p.curve.error_threshold))
        self.assertEqual(len(self.notes), 1)

    def windows_of_day_five(self):
        smooth = preprocessing.smooth(preprocessing.to_minutes(self.series)).dropna()
        return smooth.to_numpy()[4 * 1440 - 15:][:WINDOW * 2]

    def test_curve_normal_below_deformed_above_threshold(self):
        curve = self.model.profile("d1").curve
        window = self.windows_of_day_five()[:WINDOW].copy()
        _, error = curve.reconstruct(window)
        self.assertLess(error, curve.error_threshold)
        deformed = window.copy()
        deformed[80:120] = 3 * curve.training_max
        recon, error = curve.reconstruct(deformed)
        self.assertGreater(error, curve.error_threshold)
        self.assertEqual(recon.shape, (WINDOW,))

    def test_pickle_roundtrip(self):
        window = self.windows_of_day_five()[:WINDOW]
        before = self.model.profile("d1").curve.reconstruct(window)
        restored = cloudpickle.loads(cloudpickle.dumps(self.model))
        self.assertIsInstance(restored, AnomalyModel)
        curve = restored.profile("d1").curve
        self.assertIsNone(curve._module)
        after = curve.reconstruct(window)
        np.testing.assert_allclose(after[0], before[0], rtol=1e-5)
        self.assertAlmostEqual(after[1], before[1], places=5)


class Holdout(unittest.TestCase):
    """The threshold of the continuous curve comes from windows the network was not trained on."""

    def captured(self, series, log=None):
        seen = {}
        net = mock.Mock()

        def train(data, epochs):
            seen["train"] = data
            return net

        def errors(_, data):
            seen["holdout"] = data
            return np.arange(len(data), dtype=float) / len(data)

        net.state_dict.return_value = {}
        with mock.patch.object(continuous_curve, "_train_autoencoder", train), mock.patch.object(continuous_curve, "_errors", errors):
            model = continuous_curve._cont_curve(series, 1, log or mock.Mock())
        return model, seen

    def ramp(self, days):
        n = days * 1440
        return pd.Series(100.0 + np.arange(n) * 0.01, index=START + pd.to_timedelta(np.arange(n), unit="min"))

    def test_no_holdout_window_holds_a_minute_of_the_training_part(self):
        model, seen = self.captured(self.ramp(4))
        # A rising ramp stays rising through the smoothing, so a shared minute would show as overlapping values.
        self.assertLess(seen["train"].max(), seen["holdout"].min())
        n_smoothed = 4 * 1440 - 29
        cut = int(n_smoothed * 0.8)
        self.assertEqual(len(seen["holdout"]), n_smoothed - cut - WINDOW + 1)
        self.assertEqual(len(seen["train"]), len(range(0, cut - WINDOW + 1, 10)))

    def test_threshold_is_the_quantile_of_the_holdout_errors(self):
        model, seen = self.captured(self.ramp(4))
        errors = np.arange(len(seen["holdout"]), dtype=float) / len(seen["holdout"])
        self.assertAlmostEqual(model.error_threshold, float(np.quantile(errors, continuous_curve.THRESHOLD_QUANTILE)))
        self.assertEqual(continuous_curve.THRESHOLD_QUANTILE, 0.999)

    def test_three_days_leave_enough_holdout_windows(self):
        model, seen = self.captured(self.ramp(continuous_curve.MIN_CURVE_DAYS))
        self.assertIsNotNone(model)
        self.assertGreaterEqual(len(seen["holdout"]), 5 * continuous_curve.MIN_HOLDOUT_WINDOWS)

    def test_too_few_holdout_windows_give_no_model_and_a_log_line(self):
        log = mock.Mock()
        with mock.patch.object(continuous_curve, "MIN_HOLDOUT_WINDOWS", 10_000):
            model, seen = self.captured(self.ramp(4), log=log)
        self.assertIsNone(model)
        self.assertNotIn("train", seen)
        log.assert_called_once()

    def test_real_fit_threshold_is_above_the_holdout_median_and_finite(self):
        model, notes, _ = fit([FakeDataset(to_frame(day_curve(4, step_s=60)))], epochs=3)
        curve = model.profile("d1").curve
        self.assertTrue(np.isfinite(curve.error_threshold) and curve.error_threshold > 0)


class LoadDevice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = load_device_series()
        cls.model, _, cls.failed = fit([FakeDataset(to_frame(cls.series))])
        cls.profile = cls.model.profile("d1")

    def test_profile(self):
        self.assertFalse(self.failed)
        self.assertEqual(self.profile.device_type, "load_device")
        self.assertLess(self.profile.baseline, 5)
        self.assertIsInstance(self.profile.curve, LoadCurveModel)
        self.assertEqual(self.profile.curve.n_loads, 20)
        # The statistic covers all values, standby included: the mean lies far below the 800 W of a run.
        self.assertLess(self.profile.values.mean, 300)
        self.assertGreater(self.profile.values.q_high, 700)
        self.assertLess(self.profile.values.q_low, 5)
        self.assertTrue(self.profile.curve.regular)

    def runs(self):
        return preprocessing.extract_loads(preprocessing.to_minutes(self.series), self.profile.baseline)

    def test_long_run_flagged_by_length_and_normal_run_not(self):
        loads = self.runs()
        lengths = [len(x) for x in loads]
        self.assertEqual(int(np.argmax(lengths)), 10)
        curve = self.profile.curve
        self.assertEqual(curve.score(loads[10].to_numpy()), "uncontinious_device_length")
        rng = np.random.RandomState(99)
        normal = 800 + rng.normal(0, 10, 35)
        self.assertIsNone(curve.score(normal))

    def test_deformed_curve_flagged(self):
        rng = np.random.RandomState(98)
        odd = 100 + rng.normal(0, 10, 35)
        self.assertEqual(self.profile.curve.score(odd), "uncontinious_device_curve")

    def test_pickle_roundtrip(self):
        restored = cloudpickle.loads(cloudpickle.dumps(self.model))
        curve = restored.profile("d1").curve
        self.assertEqual(curve.score(self.runs()[10].to_numpy()), "uncontinious_device_length")


class LoadScoring(unittest.TestCase):
    """The robust bounds on regular runs of 36 and 37 minutes, where the MAD is close to zero."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(5)
        cls.curve = fit_load_curve([500 + rng.normal(0, 5, n) for n in [36, 37] * 10])

    def run_of(self, n, seed=6):
        return 500 + np.random.RandomState(seed).normal(0, 5, n)

    def test_fields(self):
        c = self.curve
        self.assertEqual((c.length_median, c.n_loads, c.pad_length), (36.5, 20, 37))
        self.assertEqual(c.median_curve.shape, (37,))
        self.assertTrue(0 <= c.dist_median < CURVE_MIN_TOLERANCE)

    def test_overlong_run_is_flagged_by_length_where_the_forest_stopped_at_the_training_range(self):
        self.assertEqual(self.curve.score(self.run_of(105)), "uncontinious_device_length")
        self.assertEqual(self.curve.score(self.run_of(20)), "uncontinious_device_length")

    def test_runs_in_and_slightly_beyond_the_training_range_are_normal(self):
        for n in (36, 37, 40):
            self.assertIsNone(self.curve.score(self.run_of(n)), n)

    def test_half_power_in_the_second_half_is_flagged_by_curve(self):
        run = self.run_of(36)
        run[18:] /= 2
        self.assertEqual(self.curve.score(run), "uncontinious_device_curve")

    def test_normal_run_is_not_flagged(self):
        self.assertIsNone(self.curve.score(self.run_of(37, seed=7)))

    def test_length_is_checked_before_the_curve(self):
        self.assertEqual(self.curve.score(np.full(105, 50.0)), "uncontinious_device_length")

    def test_a_dominated_median_does_not_hide_one_outlier_in_the_training(self):
        rng = np.random.RandomState(5)
        loads = [500 + rng.normal(0, 5, n) for n in [36, 37] * 10] + [500 + rng.normal(0, 5, 300)]
        self.assertEqual(fit_load_curve(loads).score(self.run_of(105)), "uncontinious_device_length")

    def test_too_few_runs_or_no_power_give_no_model(self):
        self.assertIsNone(fit_load_curve([np.full(30, 500.0)] * 14))
        self.assertIsNone(fit_load_curve([np.zeros(30)] * 20))


class RunLengthRegularity(unittest.TestCase):
    def runs(self, lengths, seed=0):
        rng = np.random.RandomState(seed)
        return [300 + rng.normal(0, 5, int(n)) for n in lengths]

    def test_tv_like_runs_are_not_judged(self):
        # Real TV: 56 % of the runs are blips of 3 minutes, the rest evenings of 90 minutes and more.
        rng = np.random.RandomState(1)
        lengths = [3] * 56 + list(rng.randint(90, 151, 44))
        curve = fit_load_curve(self.runs(lengths))
        self.assertFalse(curve.regular)
        self.assertIsNone(curve.score(self.runs([120])[0]))
        self.assertIsNone(curve.score(self.runs([600])[0]))

    def test_constant_device_keeps_a_margin(self):
        stats = extreme_value.value_stats(np.full(1000, 16.0))
        self.assertAlmostEqual(stats.upper(), 16.8)
        self.assertAlmostEqual(stats.lower(), 15.2)

    def test_two_kinds_of_runs_are_not_regular(self):
        # With a width from the MAD, 10 runs of 30 and 10 of 120 minutes counted as regular.
        self.assertFalse(fit_load_curve(self.runs([30] * 10 + [120] * 10)).regular)
        self.assertFalse(fit_load_curve(self.runs(np.linspace(10, 200, 30))).regular)

    def test_dishwasher_like_runs_keep_the_length_check(self):
        rng = np.random.RandomState(2)
        curve = fit_load_curve(self.runs(rng.randint(142, 149, 30)))
        self.assertTrue(curve.regular)
        self.assertEqual(curve.score(self.runs([240])[0]), "uncontinious_device_length")
        self.assertIsNone(curve.score(self.runs([145])[0]))


class Preprocessing(unittest.TestCase):
    def test_to_minutes_equals_the_per_second_variant(self):
        index = pd.to_datetime(["2026-01-01 00:00:07", "2026-01-01 00:00:40", "2026-01-01 00:03:10",
                                "2026-01-01 00:03:10", "2026-01-01 00:07:55", "2026-01-01 00:20:00"])
        series = pd.Series([1.0, 3.0, 8.0, 99.0, 2.0, 10.0], index=index)
        unique = series[~series.index.duplicated(keep="first")]
        old = unique.resample("s").interpolate().resample("min").asfreq().dropna()
        new = preprocessing.to_minutes(series)
        pd.testing.assert_series_equal(new, old, check_freq=False, check_names=False)

    def test_to_minutes_degenerate(self):
        self.assertTrue(preprocessing.to_minutes(pd.Series([], index=pd.DatetimeIndex([]), dtype=float)).empty)
        short = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01 00:00:10", "2026-01-01 00:00:50"]))
        self.assertTrue(preprocessing.to_minutes(short).empty)

    def test_smooth_leaves_edges_nan(self):
        smoothed = preprocessing.smooth(pd.Series(np.ones(100), index=pd.date_range("2026-01-01", periods=100, freq="min")))
        self.assertTrue(smoothed.iloc[:5].isna().all() and smoothed.iloc[-5:].isna().all())
        self.assertFalse(smoothed.iloc[30:70].isna().any())

    def test_windows(self):
        w = preprocessing.windows(np.arange(300.0))
        self.assertEqual(w.shape, (10, WINDOW))
        self.assertEqual(w[1][0], 10)
        self.assertEqual(preprocessing.windows(np.arange(10.0)).shape, (0, WINDOW))

    def test_extract_loads_drops_unfinished_run(self):
        idx = pd.date_range("2026-01-01", periods=200, freq="min")
        values = np.full(200, 2.0)
        values[20:50] = 500          # finished
        values[150:] = 500           # still running at the end
        series = pd.Series(values, index=idx)
        loads = preprocessing.extract_loads(series, 2.0)
        self.assertEqual(len(loads), 1)
        # one standby sample before the start, the run, and nothing of the ten minutes that proved the end
        self.assertEqual(loads[0].index[0], idx[19])
        self.assertEqual(loads[0].index[-1], idx[50])
        mask = preprocessing.active_mask(series, 2.0)
        self.assertTrue(mask.iloc[30] and mask.iloc[180])
        self.assertFalse(mask.iloc[100])

    def minute_series(self, values):
        return pd.Series(np.asarray(values, dtype=float), index=pd.date_range("2026-01-01", periods=len(values), freq="min"))

    def test_is_load_device_continuous_consumers(self):
        wave = 30 + 18 * np.sin(np.arange(3000) / 200)  # 12 to 48 W
        self.assertFalse(preprocessing.is_load_device(self.minute_series(wave)))
        self.assertFalse(preprocessing.is_load_device(self.minute_series(np.full(600, 16.0))))

    def test_is_load_device_mostly_idle_with_short_runs(self):
        values = np.zeros(3000)
        for start in range(100, 3000, 400):
            values[start:start + 15] = 500  # under 5 % of the time: the 95 % quantile is still 0
        series = self.minute_series(values)
        idle, work = np.quantile(series.to_numpy(), [0.05, 0.95])
        self.assertEqual(work, idle)
        self.assertTrue(preprocessing.is_load_device(series))

    def test_is_load_device_heating_element_without_standby(self):
        values = np.zeros(3000)
        values[1000:1700] = 2000
        self.assertTrue(preprocessing.is_load_device(self.minute_series(values)))

    def test_is_load_device_not_for_quantities_around_zero(self):
        rng = np.random.RandomState(3)
        self.assertFalse(preprocessing.is_load_device(self.minute_series(rng.uniform(0, 25, 3000))))  # temperature
        self.assertFalse(preprocessing.is_load_device(self.minute_series(rng.uniform(-5, 20, 3000))))
        self.assertFalse(preprocessing.is_load_device(self.minute_series(rng.uniform(-3000, -100, 3000))))

    def test_is_load_device_too_short_for_a_minute(self):
        short = pd.Series([5.0, 5.0], index=pd.to_datetime(["2026-01-01 00:00:10", "2026-01-01 00:00:50"]))
        self.assertFalse(preprocessing.is_load_device(short))


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
        model, notes, failed = fit([FakeDataset(pd.concat(shards))], curves=False)
        p = model.profile("d1")
        self.assertFalse(failed)
        self.assertEqual(p.values.n, 1440 - 3)
        # One gap is bridged by the rows without a value
        self.assertAlmostEqual(p.gaps.mean_min, (1440 - 4) / (1440 - 3 - 1) * 1.0, delta=0.01)

    def test_naive_and_aware_agree(self):
        series = day_curve(1, step_s=60)
        naive = fit([FakeDataset(to_frame(series))], curves=False)[0].profile("d1")
        aware_frame = to_frame(series)
        aware_frame["time"] = aware_frame["time"].dt.tz_localize("Europe/Berlin")
        aware = fit([FakeDataset(aware_frame)], curves=False)[0].profile("d1")
        self.assertAlmostEqual(naive.values.mean, aware.values.mean)
        self.assertEqual(naive.gaps.n, aware.gaps.n)

    def test_group_by_original_input_ids_for_operator_topics(self):
        a, b = day_curve(1, step_s=60), day_curve(1, step_s=60, seed=1) + 50
        frame = pd.concat([to_frame(a, original_input_ids="devA"), to_frame(b, original_input_ids="devB")])
        meta = [{"filterType": "OperatorId", "filterValue": "op1"}]
        model, _, _ = fit([FakeDataset(frame)], meta, curves=False)
        self.assertEqual(sorted(model.devices), ["devA", "devB"])
        self.assertGreater(model.profile("devB").values.mean, model.profile("devA").values.mean + 40)

    def test_device_id_topic_ignores_the_column(self):
        frame = to_frame(day_curve(1, step_s=60), original_input_ids="other")
        model, _, _ = fit([FakeDataset(frame)], [{"filterType": "DeviceId", "filterValue": "d1"}], curves=False)
        self.assertEqual(list(model.devices), ["d1"])

    def test_device_topic_with_several_devices_groups_by_device_column(self):
        a, b = day_curve(1, step_s=60), day_curve(1, step_s=60, seed=1) + 50
        frame = pd.concat([to_frame(a, device_id="d1"), to_frame(b, device_id="d2")])
        model, _, _ = fit([FakeDataset(frame)], [{"filterType": "DeviceId", "filterValue": "d1,d2"}], curves=False)
        self.assertEqual(sorted(model.devices), ["d1", "d2"])
        self.assertGreater(model.profile("d2").values.mean, model.profile("d1").values.mean + 40)

    def test_device_filter_value_is_normalised_to_one_device_key(self):
        frame = to_frame(day_curve(1, step_s=60))
        for raw in ("a, ", " a", "a", "a,,"):
            model, _, _ = fit([FakeDataset(frame)], [{"filterType": "DeviceId", "filterValue": raw}], curves=False)
            self.assertEqual(list(model.devices), ["a"], raw)

    def test_several_devices_without_a_column_keep_the_stripped_raw_value(self):
        model, _, _ = fit([FakeDataset(to_frame(day_curve(1, step_s=60)))],
                          [{"filterType": "DeviceId", "filterValue": " a, b "}], curves=False)
        self.assertEqual(list(model.devices), ["a, b"])

    def test_two_topics_with_the_same_device_are_merged(self):
        series = day_curve(1, step_s=60)
        first, second = series[:720], series[720:]
        meta = [{"filterType": "DeviceId", "filterValue": "d1"}, {"filterType": "DeviceId", "filterValue": "d1"}]
        model, _, _ = fit([FakeDataset(to_frame(second)), FakeDataset(to_frame(first))], meta, curves=False)
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
        model, notes, failed = fit([UnreadableDataset(), good], curves=False)
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
        self.assertEqual(GapStats(0.5, 0.0, 100, 0.5).upper(), 5)
        self.assertEqual(GapStats(1.0, 0.5, 100, 10.0).upper(), 30.0)

    def test_gap_lower_is_negative_without_regular_sending(self):
        self.assertLess(GapStats(0.5, 0.4, 100, 3.0).lower(), 0)
        self.assertGreater(GapStats(0.5, 0.01, 100, 0.5).lower(), 0)

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

    def test_epochs(self):
        self.assertEqual(continuous_curve.EPOCHS, 60)


class BimodalLoadDevice(unittest.TestCase):
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
        cls.model, _, _ = fit([FakeDataset(to_frame(pd.Series(values, index=index)))], curves=False)
        cls.profile = cls.model.profile("d1")

    def test_bounds_hold_the_upper_mode_and_leave_standby_alone(self):
        self.assertEqual(self.profile.device_type, "load_device")
        v = self.profile.values
        self.assertGreaterEqual(v.upper(), 1950)
        self.assertLess(v.upper(), 3000)
        self.assertEqual(v.lower(), 0.0)  # power is never negative


class Minimums(unittest.TestCase):
    def test_too_little_data_leaves_parts_empty(self):
        short = day_curve(0.01, step_s=60)[:20]
        p = fit([FakeDataset(to_frame(short))])[0].profile("d1")
        self.assertEqual(p.device_type, "cont_device")
        self.assertIsNone(p.values)
        self.assertIsNone(p.gaps)
        self.assertIsNone(p.curve)

    def test_two_days_are_too_few_for_a_curve(self):
        p = fit([FakeDataset(to_frame(day_curve(2, step_s=60)))])[0].profile("d1")
        self.assertIsNotNone(p.values)
        self.assertIsNone(p.curve)

    def test_too_few_loads(self):
        p = fit([FakeDataset(to_frame(load_device_series(n_runs=10, long_run=99)))])[0].profile("d1")
        self.assertEqual(p.device_type, "load_device")
        self.assertIsNotNone(p.values)
        self.assertIsNone(p.curve)

    def test_no_curves_when_not_requested(self):
        model, _, _ = fit([FakeDataset(to_frame(day_curve(4, step_s=60)))], curves=False)
        p = model.profile("d1")
        self.assertIsNotNone(p.values)
        self.assertIsNone(p.curve)
        model, _, _ = fit([FakeDataset(to_frame(load_device_series()))], curves=False)
        self.assertIsNone(model.profile("d1").curve)


if __name__ == "__main__":
    unittest.main()
