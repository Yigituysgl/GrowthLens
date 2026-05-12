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



def estimate_elasticity(bookings):
    
    df = bookings.groupby(['month', 'market', 'category']).agg(
        bookings  = ('total_bookings', 'sum'),
        avg_price = ('avg_price_eur',  'mean'),
    ).reset_index()

    df = df.sort_values(['market', 'category', 'month'])

    
    df['price_chg']   = df.groupby(['market','category'])['avg_price'].pct_change()
    df['booking_chg'] = df.groupby(['market','category'])['bookings'].pct_change()

    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    
    df = df[df['price_chg'].abs() > 0.005]
    df['elasticity'] = df['booking_chg'] / df['price_chg']

    
    low, high = df['elasticity'].quantile([0.05, 0.95])
    df = df[(df['elasticity'] >= low) & (df['elasticity'] <= high)]

    avg_elasticity = df['elasticity'].mean()

    
    by_category = df.groupby('category')['elasticity'].mean().round(3)

    return round(avg_elasticity, 3), by_category



def optimize_discount(bookings, elasticity,
                      base_price=None, base_bookings=None):

    
    if base_price is None:
        base_price    = bookings['avg_price_eur'].mean()
    if base_bookings is None:
        base_bookings = bookings['total_bookings'].mean()

    results = []
    for discount in np.arange(0, 0.51, 0.01):   # 0% to 50% in 1% steps
        disc_price   = base_price * (1 - discount)

        # Volume uplift from lower price (elasticity is negative normally)
        # I use abs() because our elasticity can be positive in synthetic data
        volume_mult  = 1 + abs(elasticity) * discount
        new_bookings = base_bookings * volume_mult
        new_revenue  = new_bookings * disc_price
        base_revenue = base_bookings * base_price
        revenue_gain = new_revenue - base_revenue

        results.append({
            'discount_pct'   : round(discount * 100, 0),
            'disc_price_eur' : round(disc_price, 2),
            'new_bookings'   : round(new_bookings, 1),
            'new_revenue_eur': round(new_revenue, 2),
            'base_revenue_eur': round(base_revenue, 2),
            'revenue_gain_eur': round(revenue_gain, 2),
        })

    df_results = pd.DataFrame(results)

    
    best_idx      = df_results['new_revenue_eur'].idxmax()
    optimal       = df_results.iloc[best_idx]

    return df_results, optimal



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
        avg_roi_pct   = ('roi_pct',              'mean'),
    ).reset_index()

    summary['total_spend']   = summary['total_spend'].round(2)
    summary['total_revenue'] = summary['total_revenue'].round(2)
    summary['avg_roi_pct']   = summary['avg_roi_pct'].round(1)

    return summary.sort_values('avg_roi_pct', ascending=False)



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

    print("\n[4] Estimating price elasticity...")
    elasticity, by_cat = estimate_elasticity(bookings)
    print(f"    Average elasticity: {elasticity}")
    print(f"    Interpretation: 1% price drop → "
          f"{abs(elasticity):.1f}% booking increase")

    print("\n[5] Optimizing discount level...")
    disc_curve, optimal = optimize_discount(bookings, elasticity)
    print(f"    Base avg price     : €{bookings['avg_price_eur'].mean():.2f}")
    print(f"    Optimal discount   : {optimal['discount_pct']:.0f}%")
    print(f"    Discounted price   : €{optimal['disc_price_eur']}")
    print(f"    Revenue gain       : €{optimal['revenue_gain_eur']:,.2f}")

    print("\n[6] Channel ROI:")
    ch_roi = get_channel_roi(mkt_roi)
    print(ch_roi.to_string(index=False))

    print("\n" + "=" * 55)
    

    return bookings, mkt_roi, elasticity, disc_curve, optimal


if __name__ == "__main__":
    main()