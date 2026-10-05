import os
import sqlite3
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules"))
from crm_retention import load_churn_dataset  # noqa: E402

CUTOFF = "2024-07-01"
FEATURE_COLUMNS = ["market", "recency_days", "frequency", "total_spend_eur", "tenure_days"]


def day(offset):
    """Date `offset` days from the cutoff, as stored in the bookings table."""
    return (pd.Timestamp(CUTOFF) + pd.Timedelta(days=offset)).date().isoformat()


def make_db(bookings):
    conn = sqlite3.connect(":memory:")
    pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "market": ["Berlin", "Paris", "Rome", "Vienna", "Prague"],
        "signup_date": ["2023-01-01 00:00:00"] * 4 + ["2024-08-01 00:00:00"],
    }).to_sql("customers", conn, index=False)
    pd.DataFrame(bookings, columns=["customer_id", "booking_date", "spend_eur"]) \
        .to_sql("bookings", conn, index=False)
    return conn


HISTORY = [
    (1, day(-10), 100.0),
    (1, day(-200), 50.0),
    (2, day(-1), 80.0),
    (3, day(-300), 60.0),
    (4, day(-5), 70.0),
    (5, day(40), 90.0),      # first booking only after the cutoff
]


class TestChurnFeatures(unittest.TestCase):

    def features(self, bookings):
        conn = make_db(bookings)
        df = load_churn_dataset(conn, CUTOFF).set_index("customer_id")
        conn.close()
        return df

    def test_features_ignore_bookings_after_cutoff(self):
        before = self.features(HISTORY)
        future = [
            (1, day(0), 999.0),      # on the cutoff day itself
            (2, day(30), 500.0),
            (3, day(179), 400.0),
            (3, day(400), 300.0),    # beyond the label window
            (4, day(90), 200.0),
        ]
        after = self.features(HISTORY + future)
        pd.testing.assert_frame_equal(before[FEATURE_COLUMNS], after[FEATURE_COLUMNS])
        # The labels must react to the same future bookings, or the test proves nothing.
        self.assertFalse(before["churned"].equals(after["churned"]))

    def test_customers_without_history_are_excluded(self):
        df = self.features(HISTORY)
        self.assertNotIn(5, df.index)
        self.assertEqual(sorted(df.index), [1, 2, 3, 4])

    def test_feature_values(self):
        row = self.features(HISTORY).loc[1]
        self.assertEqual(row["recency_days"], 10)
        self.assertEqual(row["frequency"], 2)
        self.assertEqual(row["total_spend_eur"], 150.0)

    def test_label_window_is_180_days(self):
        df = self.features(HISTORY + [(1, day(179), 10.0), (2, day(180), 10.0)])
        self.assertEqual(df.loc[1, "churned"], 0)   # booked on the last day of the window
        self.assertEqual(df.loc[2, "churned"], 1)   # first booking falls just outside it
        self.assertEqual(df.loc[3, "churned"], 1)   # no booking at all after the cutoff


if __name__ == "__main__":
    unittest.main()
