import pandas as pd
import numpy as np
import sqlite3
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "growthens.db")
SQL_PATH = os.path.join(os.path.dirname(__file__), "..", "sql", "rfm_segmentation.sql")


def get_connection():
    return sqlite3.connect(DB_PATH)



def load_rfm_data():
    conn = get_connection()
    sql = open(SQL_PATH).read()
    df = pd.read_sql(sql, conn)
    conn.close()
    return df



def get_segment_summary(df):
    summary = df.groupby('rfm_segment').agg(
        customer_count   = ('customer_id',    'count'),
        avg_spend_eur    = ('total_spend_eur', 'mean'),
        avg_bookings     = ('total_bookings',  'mean'),
        avg_recency_days = ('recency_days',    'mean'),
        total_revenue    = ('total_spend_eur', 'sum'),
    ).reset_index()

    summary['avg_spend_eur']    = summary['avg_spend_eur'].round(2)
    summary['avg_bookings']     = summary['avg_bookings'].round(1)
    summary['avg_recency_days'] = summary['avg_recency_days'].round(0).astype(int)
    summary['total_revenue']    = summary['total_revenue'].round(2)

    return summary.sort_values('avg_spend_eur', ascending=False)



def train_churn_model(df):
    # Create churn label
    df = df.copy()
    df['churned'] = (
        (df['recency_days'] > 180) &
        (df['total_bookings'] >= 2)
    ).astype(int)

    
    le = LabelEncoder()
    df['market_encoded'] = le.fit_transform(df['market'])

    
    features = ['r_score', 'f_score', 'm_score',
                'total_bookings', 'total_spend_eur',
                'recency_days', 'market_encoded']

    X = df[features]
    y = df['churned']

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        n_jobs=-1        # use all CPU cores
    )
    model.fit(X_train, y_train)

    
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    
    df['churn_probability'] = model.predict_proba(X[features])[:, 1]
    df['churn_probability'] = df['churn_probability'].round(3)

    
    importance = pd.DataFrame({
        'feature'   : features,
        'importance': model.feature_importances_.round(3)
    }).sort_values('importance', ascending=False)

    return df, auc, importance



def simulate_campaign_roi(df, open_rate=0.20, rebook_rate=0.08,
                          avg_booking_value=85, cost_per_email=0.05):

    # Target: At Risk customers with high churn probability
    at_risk = df[df['rfm_segment'].isin(['At Risk', 'Needs Attention'])].copy()

    # Sort by churn probability — target the most at-risk first
    at_risk = at_risk.sort_values('churn_probability', ascending=False)

    n_targeted       = len(at_risk)
    emails_opened    = int(n_targeted * open_rate)
    re_bookings      = int(emails_opened * rebook_rate)
    revenue_recovered = re_bookings * avg_booking_value
    campaign_cost    = n_targeted * cost_per_email
    net_gain         = revenue_recovered - campaign_cost
    roi_pct          = ((revenue_recovered - campaign_cost) / campaign_cost * 100
                        if campaign_cost > 0 else 0)
    cpa              = campaign_cost / re_bookings if re_bookings > 0 else 0

    return {
        'n_targeted'        : n_targeted,
        'emails_opened'     : emails_opened,
        're_bookings'       : re_bookings,
        'revenue_recovered' : round(revenue_recovered, 2),
        'campaign_cost'     : round(campaign_cost, 2),
        'net_gain'          : round(net_gain, 2),
        'roi_pct'           : round(roi_pct, 1),
        'cpa_eur'           : round(cpa, 2),
    }



def get_market_breakdown(df):
    breakdown = df.groupby(['market', 'rfm_segment']).agg(
        customers = ('customer_id',    'count'),
        revenue   = ('total_spend_eur','sum'),
    ).reset_index()
    breakdown['revenue'] = breakdown['revenue'].round(2)
    return breakdown



def run_all():
    print("\nGrowthLens — CRM Retention Analysis")
    print("=" * 50)

    print("\n[1] Loading RFM data from SQL...")
    df = load_rfm_data()
    print(f"    Loaded {len(df):,} customers")

    print("\n[2] Segment summary:")
    summary = get_segment_summary(df)
    print(summary.to_string(index=False))

    print("\n[3] Training churn model...")
    df, auc, importance = train_churn_model(df)
    print(f"    Model AUC: {auc:.3f}  (1.0 = perfect, 0.5 = random)")
    print("\n    Feature importance:")
    print(importance.to_string(index=False))

    print("\n[4] Campaign ROI simulation:")
    roi = simulate_campaign_roi(df)
    print(f"    Customers targeted : {roi['n_targeted']:,}")
    print(f"    Emails opened      : {roi['emails_opened']:,}")
    print(f"    Re-bookings        : {roi['re_bookings']:,}")
    print(f"    Revenue recovered  : €{roi['revenue_recovered']:,.2f}")
    print(f"    Campaign cost      : €{roi['campaign_cost']:,.2f}")
    print(f"    Net gain           : €{roi['net_gain']:,.2f}")
    print(f"    ROI                : {roi['roi_pct']}%")
    print(f"    CPA                : €{roi['cpa_eur']}")

    print("\n" + "=" * 50)
    print("Done! crm_retention.py is ready.")
    return df, summary, auc, importance, roi


if __name__ == "__main__":
    run_all()