import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules"))
import pricing_opt as po  # noqa: E402

PRICE, VOLUME = 100.0, 1000.0


def best(elasticity, margin=0.20):
    _, best_rev, best_profit = po.optimize_discount(PRICE, VOLUME, elasticity, margin)
    return best_rev['discount_pct'], best_profit['discount_pct']


class TestRevenueOptimum(unittest.TestCase):

    def test_zero_discount_when_elasticity_at_most_one(self):
        for e in (0.0, 0.3, 0.8, 1.0):
            self.assertEqual(best(e)[0], 0, f"elasticity {e}")

    def test_matches_formula_above_one(self):
        # Revenue (1 + e*d)(1 - d) peaks at d = (e - 1) / (2e).
        for e in (1.5, 2.0, 3.0, 5.0):
            self.assertAlmostEqual(best(e)[0], (e - 1) / (2 * e) * 100, delta=1.0)


class TestProfitOptimum(unittest.TestCase):

    def test_zero_discount_when_elasticity_times_margin_at_most_one(self):
        for e, m in ((1.0, 0.20), (4.0, 0.20), (5.0, 0.20), (2.0, 0.50)):
            self.assertEqual(best(e, m)[1], 0, f"elasticity {e}, margin {m}")

    def test_matches_formula_when_discount_pays(self):
        # Profit (1 + e*d)(m - d) peaks at d = m/2 - 1/(2e).
        for e, m in ((10.0, 0.20), (5.0, 0.50), (4.0, 0.60)):
            self.assertAlmostEqual(best(e, m)[1], (m / 2 - 1 / (2 * e)) * 100, delta=1.0)

    def test_profit_optimum_never_above_revenue_optimum(self):
        for e in (0.5, 1.5, 3.0, 5.0):
            for m in (0.1, 0.3, 0.6):
                rev, prof = best(e, m)
                self.assertLessEqual(prof, rev)


class TestCurveValues(unittest.TestCase):

    def test_hand_calculated_point(self):
        curve, _, _ = po.optimize_discount(PRICE, VOLUME, elasticity=2.0, margin=0.30)
        row = curve.set_index('discount_pct').loc[10.0]
        self.assertAlmostEqual(row['bookings'], 1200)              # 1,000 x (1 + 2 x 0.10)
        self.assertAlmostEqual(row['revenue_eur'], 108_000)        # 1,200 x €90
        self.assertAlmostEqual(row['profit_eur'], 24_000)          # 1,200 x €100 x (0.30 - 0.10)
        self.assertAlmostEqual(row['profit_change_eur'], -6_000)   # vs 1,000 x €30

    def test_discount_above_margin_loses_money_per_booking(self):
        curve, _, _ = po.optimize_discount(PRICE, VOLUME, elasticity=3.0, margin=0.20)
        self.assertTrue((curve.loc[curve['discount_pct'] > 20, 'profit_eur'] < 0).all())

    def test_negative_elasticity_is_rejected_not_flipped(self):
        with self.assertRaises(ValueError):
            po.optimize_discount(PRICE, VOLUME, elasticity=-0.5, margin=0.2)

    def test_estimated_elasticity_is_gone(self):
        self.assertFalse(hasattr(po, 'estimate_elasticity'))


class TestExplanationAndBase(unittest.TestCase):

    def test_explanation_gives_reason_for_zero(self):
        _, br, bp = po.optimize_discount(PRICE, VOLUME, 0.8, 0.20)
        text = " ".join(po.explain_discount(0.8, 0.20, br, bp))
        self.assertIn("0% maximises revenue", text)
        self.assertIn("0% maximises profit", text)

    def test_base_values_are_booking_weighted(self):
        bookings = pd.DataFrame({
            'month'            : ['2024-01', '2024-01', '2024-02'],
            'total_bookings'   : [9, 1, 10],
            'total_revenue_eur': [900.0, 500.0, 1000.0],
        })
        price, monthly = po.get_base_values(bookings)
        self.assertAlmostEqual(price, 2400 / 20)    # not the mean of 100, 500 and 100
        self.assertAlmostEqual(monthly, 10)


if __name__ == "__main__":
    unittest.main()
