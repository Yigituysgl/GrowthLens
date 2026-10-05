import pandas as pd
import numpy as np
import sqlite3
import os

DB_PATH       = os.path.join(os.path.dirname(__file__), "..", "data", "growthens.db")
SQL_PRICING   = os.path.join(os.path.dirname(__file__), "..", "sql", "pricing_analysis.sql")
SQL_MKT_ROI   = os.path.join(os.path.dirname(__file__), "..", "sql", "marketing_roi.sql")


def get_connection():
    return sqlite3.connect(DB_PATH)



def load_data():
    conn     = get_connection()
    bookings = pd.read_sql(open(SQL_PRICING).read(),  conn)
    mkt_roi  = pd.read_sql(open(SQL_MKT_ROI).read(),  conn)
    conn.close()
    return bookings, mkt_roi



ELASTICITY_NOTE = (
    "In this synthetic data, the number of bookings does not depend on price, "
    "so price elasticity cannot be estimated from it. Elasticity is therefore "
    "an assumption you set."
)

PRICE_TEST_NOTE = (
    "Real elasticity must be measured with a price A/B test: show different "
    "prices to randomly chosen groups and compare bookings."
)


def get_base_values(bookings):
    # Weighted by bookings; averaging the per-group average prices would give
    # small groups the same weight as large ones.
    base_price = bookings['total_revenue_eur'].sum() / bookings['total_bookings'].sum()
    base_monthly_bookings = bookings['total_bookings'].sum() / bookings['month'].nunique()
    return base_price, base_monthly_bookings



def optimize_discount(base_price, base_bookings, elasticity, margin):
    # elasticity = % more bookings per 1% price cut, entered as a positive number.
    if elasticity < 0:
        raise ValueError("elasticity must be >= 0 (bookings rise when price falls)")

    discount = np.round(np.arange(0, 0.51, 0.01), 2)     # 0% to 50% in 1% steps
    # Straight-line demand response; only a fair approximation for modest discounts.
    bookings_at = base_bookings * (1 + elasticity * discount)
    revenue     = bookings_at * base_price * (1 - discount)
    # Cost per booking is fixed at price x (1 - margin), so the discount comes
    # straight out of the margin.
    profit      = bookings_at * base_price * (margin - discount)

    curve = pd.DataFrame({
        'discount_pct'  : discount * 100,
        'disc_price_eur': base_price * (1 - discount),
        'bookings'      : bookings_at,
        'revenue_eur'   : revenue,
        'profit_eur'    : profit,
    })
    curve['revenue_change_eur'] = curve['revenue_eur'] - curve['revenue_eur'].iloc[0]
    curve['profit_change_eur']  = curve['profit_eur']  - curve['profit_eur'].iloc[0]

    # idxmax returns the first maximum, so ties resolve to the smaller discount.
    best_revenue = curve.loc[curve['revenue_eur'].idxmax()]
    best_profit  = curve.loc[curve['profit_eur'].idxmax()]
    return curve, best_revenue, best_profit



def format_eur_change(x):
    return f"{'-' if x < 0 else '+'}€{abs(x):,.0f}"



def explain_discount(elasticity, margin, best_revenue, best_profit):
    lines = []
    if elasticity <= 1:
        lines.append(
            f"Each 1% off the price brings only {elasticity:.1f}% more bookings, so the extra "
            f"volume never makes up for the lower price. That is why 0% maximises revenue."
        )
    else:
        lines.append(
            f"Each 1% off the price brings {elasticity:.1f}% more bookings, so a discount "
            f"can grow revenue; it peaks at {best_revenue['discount_pct']:.0f}%."
        )
    if elasticity * margin <= 1:
        lines.append(
            f"Profit is stricter: with a {margin*100:.0f}% margin, each 1% off the price "
            f"removes {1/margin:.1f}% of the profit on every booking, so bookings must rise "
            f"by more than {1/margin:.1f}% per 1% cut to pay for it. At elasticity "
            f"{elasticity:.1f} they do not, so 0% maximises profit."
        )
    else:
        lines.append(
            f"With a {margin*100:.0f}% margin the extra volume outweighs the thinner margin; "
            f"profit peaks at a discount of {best_profit['discount_pct']:.0f}% "
            f"({format_eur_change(best_profit['profit_change_eur'])} vs no discount)."
        )
    return lines



def get_revenue_summary(bookings):
    by_market = bookings.groupby('market').agg(
        total_bookings  = ('total_bookings',   'sum'),
        total_revenue   = ('total_revenue_eur','sum'),
        avg_price       = ('avg_price_eur',    'mean'),
    ).reset_index().sort_values('total_revenue', ascending=False)

    by_category = bookings.groupby('category').agg(
        total_bookings  = ('total_bookings',   'sum'),
        total_revenue   = ('total_revenue_eur','sum'),
        avg_price       = ('avg_price_eur',    'mean'),
    ).reset_index().sort_values('total_revenue', ascending=False)

    by_market['total_revenue']   = by_market['total_revenue'].round(2)
    by_market['avg_price']       = by_market['avg_price'].round(2)
    by_category['total_revenue'] = by_category['total_revenue'].round(2)
    by_category['avg_price']     = by_category['avg_price'].round(2)

    return by_market, by_category



def get_channel_roi(mkt_roi):
    summary = mkt_roi.groupby('channel').agg(
        total_spend   = ('total_marketing_spend','sum'),
        total_revenue = ('total_revenue',        'sum'),
    ).reset_index()

    # ROI from channel totals; averaging market ROIs would weight a small
    # market the same as a large one.
    spend = summary['total_spend'].where(summary['total_spend'] > 0)
    summary['roi_pct'] = ((summary['total_revenue'] - spend) / spend * 100).round(1)

    summary['total_spend']   = summary['total_spend'].round(2)
    summary['total_revenue'] = summary['total_revenue'].round(2)

    # A channel without spend has no ROI, so it goes last rather than first.
    return summary.sort_values('roi_pct', ascending=False, na_position='last')



def format_roi(roi_pct):
    return "n/a (no spend)" if pd.isna(roi_pct) else f"{roi_pct:,.1f}%"



def main():
    print("\nGrowthLens — Pricing Optimization Analysis")
    print("=" * 55)

    print("\n[1] Loading data from SQL...")
    bookings, mkt_roi = load_data()
    print(f"    Booking rows : {len(bookings):,}")
    print(f"    Marketing rows: {len(mkt_roi):,}")

    print("\n[2] Revenue by market:")
    by_market, by_category = get_revenue_summary(bookings)
    print(by_market.to_string(index=False))

    print("\n[3] Revenue by category (top 5):")
    print(by_category.head(5).to_string(index=False))

    print("\n[4] Price elasticity:")
    print(f"    {ELASTICITY_NOTE}")

    elasticity, margin = 1.0, 0.20
    print(f"\n[5] Discount optimizer (assumed elasticity {elasticity}, margin {margin:.0%}):")
    base_price, base_bookings = get_base_values(bookings)
    curve, best_rev, best_profit = optimize_discount(base_price, base_bookings,
                                                     elasticity, margin)
    print(f"    Base price (data)          : €{base_price:.2f}")
    print(f"    Base monthly bookings (data): {base_bookings:,.0f}")
    print(f"    Revenue-optimal discount   : {best_rev['discount_pct']:.0f}%")
    print(f"    Profit-optimal discount    : {best_profit['discount_pct']:.0f}%")
    for line in explain_discount(elasticity, margin, best_rev, best_profit):
        print(f"    - {line}")
    print(f"    {PRICE_TEST_NOTE}")

    print("\n[6] Channel ROI:")
    ch_roi = get_channel_roi(mkt_roi)
    print(ch_roi.to_string(index=False))

    print("\n" + "=" * 55)
    

    return bookings, mkt_roi, curve, best_rev, best_profit


if __name__ == "__main__":
    main()