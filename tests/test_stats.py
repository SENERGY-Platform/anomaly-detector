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

import unittest

import numpy as np

from algo.stats import (FALLBACK_SAMPLES, MIN_SAMPLES, STATS_REFRESH, STATS_REFRESH_ALWAYS_BELOW, GapStats,
                        SampleBuffer, ValueStats, gap_stats, value_stats)


class TestValueStats(unittest.TestCase):
    def test_the_bounds_are_widened_by_half_their_distance(self):
        self.assertEqual((ValueStats(10.0, -10.0, 90.0).lower(), ValueStats(10.0, -10.0, 90.0).upper()),
                         (-60.0, 140.0))

    def test_the_lower_bound_of_a_never_negative_quantity_stops_at_zero(self):
        stats = ValueStats(10.0, 0.0, 100.0)
        self.assertEqual((stats.lower(), stats.upper()), (0.0, 150.0))

    def test_a_device_with_a_base_load_has_a_lower_bound_above_zero(self):
        self.assertEqual(ValueStats(60.0, 40.0, 80.0).lower(), 20.0)

    def test_a_constant_device_keeps_a_margin_of_five_percent(self):
        stats = ValueStats(100.0, 100.0, 100.0)
        self.assertEqual((stats.lower(), stats.upper()), (95.0, 105.0))

    def test_the_bounds_come_from_the_ends_of_the_values(self):
        stats = value_stats(np.tile([65.0, 135.0], 500))
        self.assertEqual(stats, ValueStats(100.0, 65.0, 135.0))

    def test_the_five_most_extreme_values_on_each_side_are_ignored(self):
        stats = value_stats(np.r_[np.full(5, -1e4), np.linspace(0.0, 100.0, 990), np.full(5, 1e4)])
        self.assertEqual((stats.q_low, stats.q_high), (0.0, 100.0))

    def test_a_short_history_ignores_at_most_one_value_in_two_hundred(self):
        stats = value_stats(np.r_[np.full(2, -1e4), np.linspace(0.0, 100.0, 396), np.full(2, 1e4)])
        self.assertEqual((stats.q_low, stats.q_high), (0.0, 100.0))

    def test_fewer_than_two_hundred_values_keep_their_extremes(self):
        stats = value_stats(np.r_[np.linspace(0.0, 100.0, 198), 1e4])
        self.assertEqual((stats.q_low, stats.q_high), (0.0, 1e4))

    def test_an_outlier_in_the_history_does_not_hide_the_same_outlier_again(self):
        # 50 days of a device that sends every hour, with one outlier among them.
        stats = value_stats(np.r_[np.linspace(0.0, 1000.0, 1199), 1600.0])
        self.assertLess(stats.upper(), 1600.0)

    def test_too_few_values_give_no_statistic(self):
        self.assertIsNone(value_stats(np.ones(MIN_SAMPLES - 1)))
        self.assertIsNotNone(value_stats(np.ones(MIN_SAMPLES)))


class TestGapStats(unittest.TestCase):
    def test_the_bound_is_three_times_the_999_quantile_and_at_least_five_minutes(self):
        self.assertEqual(GapStats(0.5, 0.5).upper(), 5)
        self.assertEqual(GapStats(1.0, 10.0).upper(), 30.0)

    def test_too_few_gaps_give_no_statistic(self):
        self.assertIsNone(gap_stats(np.ones(MIN_SAMPLES - 1)))
        self.assertEqual(gap_stats(np.full(MIN_SAMPLES, 2.0)), GapStats(2.0, 2.0))


class TestSampleBuffer(unittest.TestCase):
    def filled(self, values):
        buffer = SampleBuffer(value_stats)
        for v in values:
            buffer.add(float(v))
        return buffer

    def test_a_young_statistic_follows_every_value(self):
        buffer = self.filled(range(MIN_SAMPLES))
        self.assertLess(buffer.stats().q_high, MIN_SAMPLES)
        buffer.add(1000.0)
        self.assertGreater(buffer.stats().q_high, MIN_SAMPLES)

    def test_an_old_statistic_is_recomputed_every_stats_refresh_values(self):
        buffer = self.filled(np.arange(STATS_REFRESH_ALWAYS_BELOW) % 30)
        first = buffer.stats()
        for _ in range(STATS_REFRESH - 1):
            buffer.add(1000.0)
        self.assertIs(buffer.stats(), first)
        buffer.add(1000.0)
        self.assertEqual(buffer.stats().q_high, 1000.0)

    def test_it_forgets_what_lies_beyond_the_last_fallback_samples(self):
        buffer = self.filled(np.r_[np.full(STATS_REFRESH, 1e6), np.full(FALLBACK_SAMPLES, 100.0)])
        self.assertEqual(buffer.stats().q_high, 100.0)

    def test_the_live_statistic_is_the_statistic_of_the_history(self):
        values = np.random.RandomState(3).gamma(2.0, 30.0, 500)
        self.assertEqual(self.filled(values).stats(), value_stats(values))


if __name__ == "__main__":
    unittest.main()
