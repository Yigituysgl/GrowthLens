import os
import sqlite3
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules"))
from pricing_opt import format_roi, get_channel_roi  # noqa: E402

ROOT     = os.path.join(os.path.dirname(__file__), "..")
SQL_PATH = os.path.join(ROOT, "sql", "marketing_roi.sql")
DB_PATH  = os.path.join(ROOT, "data", "growthens.db")


def run_roi(conn):
    with open(SQL_PATH) as f:
        return pd.read_sql(f.read(), conn)


def make_db(spend_rows, booking_rows):
    conn = sqlite3.connect(":memory:")
    pd.DataFrame(spend_rows, columns=["period", "market", "channel", "spend_eur"]) \
        .to_sql("marketing_spend", conn, index=False)
    pd.DataFrame(booking_rows, columns=["customer_id", "booking_date", "market",
                                        "channel", "spend_eur"]) \
        .to_sql("bookings", conn, index=False)
    return conn


# Several spend months and several bookings per market/channel: the old
# row-level join multiplied both sides here.
SPEND = [
    ("2024-01", "Berlin", "Paid Search", 1000.0),
    ("2024-02", "Berlin", "Paid Search", 1000.0),
    ("2024-03", "Berlin", "Paid Search", 1000.0),
    ("2024-01", "Berlin", "CRM Email",    200.0),
    ("2024-02", "Berlin", "CRM Email",    300.0),
    ("2024-01", "Paris",  "Paid Search",  500.0),
    ("2024-02", "Paris",  "Paid Search",  500.0),
]
BOOKINGS = [
    (1, "2024-01-05", "Berlin", "Paid Search", 1500.0),
    (2, "2024-02-05", "Berlin", "Paid Search", 1500.0),
    (3, "2024-03-05", "Berlin", "Paid Search", 1500.0),
    (4, "2024-01-09", "Berlin", "CRM Email",    400.0),
    (5, "2024-02-09", "Berlin", "CRM Email",    600.0),
    (6, "2024-01-15", "Paris",  "Paid Search",  800.0),
]


class TestMarketingRoiTotals(unittest.TestCase):

    def check_totals(self, conn):
        roi = run_roi(conn)
        raw_spend   = pd.read_sql("SELECT SUM(spend_eur) s FROM marketing_spend", conn).s[0]
        raw_revenue = pd.read_sql("SELECT SUM(spend_eur) s FROM bookings", conn).s[0]
        raw_count   = pd.read_sql("SELECT COUNT(*) n FROM bookings", conn).n[0]
        self.assertAlmostEqual(roi['total_marketing_spend'].sum(), raw_spend, places=0)
        self.assertAlmostEqual(roi['total_revenue'].sum(), raw_revenue, places=0)
        self.assertEqual(roi['total_bookings'].sum(), raw_count)
        return roi

    def test_totals_match_source_tables(self):
        self.check_totals(make_db(SPEND, BOOKINGS))

    def test_roi_matches_hand_calculation(self):
        roi = run_roi(make_db(SPEND, BOOKINGS)).set_index(["market", "channel"])
        # Berlin Paid Search: revenue 4,500, spend 3,000 -> +50%
        self.assertAlmostEqual(roi.loc[("Berlin", "Paid Search"), "roi_pct"], 50.0)
        # Berlin CRM Email: revenue 1,000, spend 500 -> +100%
        self.assertAlmostEqual(roi.loc[("Berlin", "CRM Email"), "roi_pct"], 100.0)

    def test_channel_roi_uses_totals(self):
        ch = get_channel_roi(run_roi(make_db(SPEND, BOOKINGS))).set_index("channel")
        # Paid Search: revenue 4,500 + 800 = 5,300, spend 3,000 + 1,000 = 4,000 -> +32.5%
        self.assertAlmostEqual(ch.loc["Paid Search", "roi_pct"], 32.5)

    @unittest.skipUnless(os.path.exists(DB_PATH), "project database not present")
    def test_totals_match_on_project_database(self):
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        try:
            self.check_totals(conn)
        finally:
            conn.close()


class TestZeroSpendChannel(unittest.TestCase):

    def setUp(self):
        spend = SPEND + [("2024-01", "Berlin", "Organic", 0.0),
                         ("2024-02", "Berlin", "Organic", 0.0)]
        bookings = BOOKINGS + [(7, "2024-01-20", "Berlin", "Organic", 900.0)]
        self.roi = run_roi(make_db(spend, bookings))
        self.channels = get_channel_roi(self.roi)

    def test_zero_spend_has_no_roi(self):
        organic = self.roi[self.roi['channel'] == "Organic"].iloc[0]
        self.assertTrue(pd.isna(organic['roi_pct']))
        self.assertTrue(pd.isna(
            self.channels.set_index("channel").loc["Organic", "roi_pct"]))

    def test_zero_spend_channel_is_not_ranked_best(self):
        self.assertNotEqual(self.channels.iloc[0]['channel'], "Organic")
        self.assertEqual(self.channels.iloc[-1]['channel'], "Organic")

    def test_zero_spend_is_shown_as_not_applicable(self):
        self.assertEqual(format_roi(float('nan')), "n/a (no spend)")
        self.assertEqual(format_roi(32.5), "32.5%")


if __name__ == "__main__":
    unittest.main()
