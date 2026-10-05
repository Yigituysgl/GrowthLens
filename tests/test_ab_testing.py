import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules"))
import ab_testing as ab  # noqa: E402
import data_generator as gen  # noqa: E402


class TestConfidenceInterval(unittest.TestCase):

    def test_interval_contains_point_estimate(self):
        ci = ab.lift_confidence_interval(100, 5000, 150, 5000)
        self.assertLess(ci['diff_low'], ci['diff'])
        self.assertGreater(ci['diff_high'], ci['diff'])
        self.assertLess(ci['rel_low'], ci['rel_lift'])
        self.assertGreater(ci['rel_high'], ci['rel_lift'])

    def test_relative_interval_matches_hand_calculation(self):
        # 2% vs 3%: log(1.5) ± 1.96 * sqrt(1/150 - 1/5000 + 1/100 - 1/5000)
        ci = ab.lift_confidence_interval(100, 5000, 150, 5000)
        self.assertAlmostEqual(ci['rel_lift'], 0.5)
        self.assertAlmostEqual(ci['rel_low'], 0.1682, places=3)
        self.assertAlmostEqual(ci['rel_high'], 0.9260, places=3)

    def test_no_effect_interval_includes_zero(self):
        ci = ab.lift_confidence_interval(100, 5000, 100, 5000)
        self.assertLess(ci['diff_low'], 0)
        self.assertGreater(ci['diff_high'], 0)
        self.assertLess(ci['rel_low'], 0)
        self.assertGreater(ci['rel_high'], 0)


class TestPower(unittest.TestCase):

    def test_power_grows_with_sample_size(self):
        powers = [ab.power_two_proportions(0.02, 0.03, n, n) for n in (500, 2000, 8000)]
        self.assertEqual(powers, sorted(powers))
        self.assertGreater(powers[-1], 0.95)

    def test_sample_size_gives_target_power(self):
        for target in (0.80, 0.90):
            n = ab.sample_size_per_group(0.018, 0.026, target)
            self.assertGreaterEqual(ab.power_two_proportions(0.018, 0.026, n, n), target)
            self.assertLess(ab.power_two_proportions(0.018, 0.026, n - 50, n - 50), target)

    def test_power_matches_simulation(self):
        p_c, p_t, n, runs = 0.02, 0.03, 2000, 4000
        rng = np.random.default_rng(7)
        conv_c = rng.binomial(n, p_c, runs)
        conv_t = rng.binomial(n, p_t, runs)
        hits = sum(ab._ztest_proportions(c, n, t, n)[1] < 0.05 for c, t in zip(conv_c, conv_t))
        self.assertAlmostEqual(hits / runs, ab.power_two_proportions(p_c, p_t, n, n), delta=0.03)


class TestSampleRatio(unittest.TestCase):

    def test_even_split_passes(self):
        self.assertTrue(ab.sample_ratio_check(5000, 5000)['srm_ok'])

    def test_uneven_split_is_flagged(self):
        r = ab.sample_ratio_check(5000, 4500)
        self.assertFalse(r['srm_ok'])
        self.assertLess(r['srm_p_value'], 0.001)


class TestPlantedTruth(unittest.TestCase):

    def test_generator_uses_the_planted_rates(self):
        # Extreme rates make the outcome certain, proving the generator reads the constant.
        extreme = {"urgency_banner": {"control": 0.0, "test": 1.0},
                   "discount_email": {"control": 1.0, "test": 0.0}}
        customers = pd.DataFrame({"customer_id": range(1, 12_001), "email_opt_in": 1})
        with mock.patch.dict(gen.PLANTED_CONVERSION, extreme):
            exp = gen.generate_experiments(customers)
        rates = exp.groupby(["experiment", "variant"])["converted"].mean()
        for name, by_variant in extreme.items():
            for variant, rate in by_variant.items():
                self.assertEqual(rates[(name, variant)], rate)

    def test_module_compares_against_generator_rates(self):
        self.assertIs(ab.PLANTED_CONVERSION, gen.PLANTED_CONVERSION)

    def test_missed_effect_is_flagged(self):
        ci = ab.lift_confidence_interval(94, 4980, 120, 5020)
        truth = ab.compare_with_truth("urgency_banner", 4980, 5020, ci, is_significant=False)
        self.assertTrue(truth['missed_real_effect'])
        self.assertFalse(truth['verdict_correct'])
        self.assertLess(truth['power_at_true_effect'], 0.80)


if __name__ == "__main__":
    unittest.main()
