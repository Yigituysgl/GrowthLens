import math
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules"))
from crm_retention import (baseline_rebook_rate, campaign_economics,  # noqa: E402
                           select_campaign_targets)

# 1,000 targeted, 60% re-book anyway, +5 pp uplift, €100 bookings, 20% margin, €0.05/email.
BASE = dict(n_targeted=1000, baseline_rate=0.60, uplift=0.05,
            avg_booking_value=100.0, margin=0.20, cost_per_email=0.05)


class TestCampaignEconomics(unittest.TestCase):

    def test_hand_calculated_example(self):
        r = campaign_economics(**BASE)
        self.assertAlmostEqual(r['would_book_anyway'], 600)
        self.assertAlmostEqual(r['extra_rebookings'], 50)       # 1,000 x 5 pp
        self.assertAlmostEqual(r['extra_revenue'], 5_000)       # 50 x €100
        self.assertAlmostEqual(r['extra_profit'], 1_000)        # 20% of €5,000
        self.assertAlmostEqual(r['email_cost'], 50)             # 1,000 x €0.05
        self.assertAlmostEqual(r['net_profit'], 950)
        self.assertAlmostEqual(r['roi_pct'], 1_900)             # 950 / 50

    def test_roi_uses_profit_not_revenue(self):
        r = campaign_economics(**BASE)
        revenue_roi = (r['extra_revenue'] - r['total_cost']) / r['total_cost'] * 100
        self.assertNotAlmostEqual(r['roi_pct'], revenue_roi)
        self.assertAlmostEqual(r['roi_pct'],
                               (r['extra_profit'] - r['total_cost']) / r['total_cost'] * 100)

    def test_zero_uplift_loses_the_whole_cost(self):
        r = campaign_economics(**{**BASE, 'uplift': 0.0})
        self.assertEqual(r['extra_rebookings'], 0)
        self.assertAlmostEqual(r['net_profit'], -50)
        self.assertAlmostEqual(r['roi_pct'], -100)

    def test_customers_who_book_anyway_are_not_credited(self):
        low = campaign_economics(**{**BASE, 'baseline_rate': 0.10})
        high = campaign_economics(**{**BASE, 'baseline_rate': 0.60})
        self.assertAlmostEqual(low['extra_rebookings'], high['extra_rebookings'])
        self.assertAlmostEqual(low['net_profit'], high['net_profit'])

    def test_extra_rebookings_capped_by_customers_left_to_win(self):
        r = campaign_economics(**{**BASE, 'baseline_rate': 0.98})
        self.assertAlmostEqual(r['extra_rebookings'], 20)       # only 2% did not book anyway

    def test_discount_cost_includes_customers_who_book_anyway(self):
        r = campaign_economics(**BASE, discount=0.10)
        self.assertAlmostEqual(r['anyway_discount_cost'], 6_000)   # 600 x €100 x 10%
        self.assertAlmostEqual(r['extra_revenue'], 4_500)          # 50 x €90
        self.assertAlmostEqual(r['extra_profit'], 500)             # 50 x €100 x (20% - 10%)
        self.assertAlmostEqual(r['total_cost'], 6_050)
        self.assertAlmostEqual(r['net_profit'], -5_550)
        self.assertAlmostEqual(r['roi_pct'], -5_550 / 6_050 * 100)

    def test_no_discount_means_no_discount_cost(self):
        self.assertEqual(campaign_economics(**BASE)['anyway_discount_cost'], 0)

    def test_no_cost_gives_undefined_roi(self):
        r = campaign_economics(**{**BASE, 'cost_per_email': 0.0})
        self.assertTrue(math.isnan(r['roi_pct']))


class TestTargeting(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'customer_id'      : [1, 2, 3, 4, 5, 6],
            'email_opt_in'     : [1, 1, 0, 1, 1, 1],
            'churn_probability': [0.9, 0.2, 0.99, float('nan'), 0.5, 0.7],
            'churned'          : [1, 0, 1, 0, 0, 1],
        })

    def test_targets_only_opted_in_scored_customers_highest_risk_first(self):
        targets = select_campaign_targets(self.df, 0.5)
        self.assertEqual(targets['customer_id'].tolist(), [1, 6])   # 2 of 4 eligible

    def test_full_target_excludes_opted_out_and_unscored(self):
        targets = select_campaign_targets(self.df, 1.0)
        self.assertEqual(sorted(targets['customer_id']), [1, 2, 5, 6])

    def test_baseline_rate_comes_from_the_same_selection(self):
        # Top half = customers 1 and 6, both churned, so nobody re-booked.
        self.assertEqual(baseline_rebook_rate(self.df, 0.5), 0.0)
        # All four eligible: two churned, two re-booked.
        self.assertEqual(baseline_rebook_rate(self.df, 1.0), 0.5)


if __name__ == "__main__":
    unittest.main()
