import os
import sqlite3
import unittest

import pandas as pd

SQL_PATH = os.path.join(os.path.dirname(__file__), "..", "sql", "rfm_segmentation.sql")

# The SQL measures recency against this fixed end date of the dataset.
END_DATE = pd.Timestamp("2024-12-31")


def run_rfm(bookings_by_customer):
    """Run the real RFM query against a small in-memory database.

    bookings_by_customer maps customer_id -> list of (days_before_end, spend_eur).
    """
    conn = sqlite3.connect(":memory:")
    customers = pd.DataFrame({
        "customer_id": list(bookings_by_customer),
        "market": "Berlin",
        "segment": "Regular",
    })
    rows = [
        {"customer_id": cid,
         "booking_date": (END_DATE - pd.Timedelta(days=days)).date().isoformat(),
         "spend_eur": spend}
        for cid, bookings in bookings_by_customer.items()
        for days, spend in bookings
    ]
    bookings = pd.DataFrame(rows, columns=["customer_id", "booking_date", "spend_eur"])
    customers.to_sql("customers", conn, index=False)
    bookings.to_sql("bookings", conn, index=False)
    with open(SQL_PATH) as f:
        df = pd.read_sql(f.read(), conn)
    conn.close()
    return df.set_index("customer_id")


class TestRfmSegmentation(unittest.TestCase):

    def setUp(self):
        self.df = run_rfm({
            1: [(5, 100)],                         # booked 5 days before the end
            2: [(30, 100)],
            3: [(200, 100)],
            4: [(600, 100)],                       # booked long ago
            5: [],                                 # never booked
            6: [(10, 50), (40, 50)],               # two bookings
            7: [(300, 80), (320, 80)],             # two bookings
        })

    def test_recent_booker_gets_higher_recency_score(self):
        self.assertGreater(self.df.loc[1, "r_score"], self.df.loc[4, "r_score"])

    def test_recency_score_never_rises_as_bookings_get_older(self):
        bookers = self.df.dropna(subset=["r_score"]).sort_values("recency_days")
        self.assertTrue(bookers["r_score"].is_monotonic_decreasing)

    def test_never_booked_has_own_segment_and_no_scores(self):
        row = self.df.loc[5]
        self.assertEqual(row["rfm_segment"], "Never booked")
        self.assertEqual(row["total_bookings"], 0)
        self.assertTrue(pd.isna(row["r_score"]))
        self.assertTrue(pd.isna(row["f_score"]))
        self.assertTrue(pd.isna(row["m_score"]))

    def test_same_booking_count_gets_same_frequency_score(self):
        self.assertEqual(self.df.loc[6, "f_score"], self.df.loc[7, "f_score"])
        self.assertEqual(self.df.loc[1, "f_score"], self.df.loc[4, "f_score"])

    def test_recent_single_booking_is_recent_one_time_buyer(self):
        self.assertEqual(self.df.loc[1, "rfm_segment"], "Recent one-time buyer")

    def test_old_single_booking_is_lost(self):
        self.assertEqual(self.df.loc[4, "rfm_segment"], "Lost")


if __name__ == "__main__":
    unittest.main()
