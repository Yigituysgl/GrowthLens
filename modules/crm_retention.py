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
CHURN_SQL_PATH = os.path.join(os.path.dirname(__file__), "..", "sql", "churn_features.sql")

# Bookings run to 2024-12-31; this cutoff leaves a full 180-day window to observe churn.
TRAIN_CUTOFF       = "2024-07-01"
CHURN_HORIZON_DAYS = 180
# Day after the last booking, so current risk scores use the full history.
SCORING_DATE       = "2025-01-01"

# customers.segment is left out: it is the generator's hidden booking-rate group.
MODEL_FEATURES = ['recency_days', 'frequency', 'total_spend_eur',
                  'tenure_days', 'market_encoded']


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



def load_churn_dataset(conn, cutoff, horizon_days=CHURN_HORIZON_DAYS):
    label_end = (pd.Timestamp(cutoff) + pd.Timedelta(days=horizon_days)).date().isoformat()
    with open(CHURN_SQL_PATH) as f:
        sql = f.read()
    return pd.read_sql(sql, conn, params={'cutoff': cutoff, 'label_end': label_end})



def top_decile_capture(y_true, score):
    # Stable sort keeps tie-breaking reproducible between runs.
    k = max(1, int(round(len(score) * 0.10)))
    top = np.argsort(-np.asarray(score, dtype=float), kind='stable')[:k]
    return int(np.asarray(y_true)[top].sum()), k



def train_churn_model(rfm_df):
    conn = get_connection()
    train_set = load_churn_dataset(conn, TRAIN_CUTOFF)
    # Labels at the scoring date are meaningless (no future data); only features are used.
    current   = load_churn_dataset(conn, SCORING_DATE)
    conn.close()

    le = LabelEncoder().fit(train_set['market'])
    train_set['market_encoded'] = le.transform(train_set['market'])
    current['market_encoded']   = le.transform(current['market'])

    X = train_set[MODEL_FEATURES]
    y = train_set['churned']
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

    # Baseline: longer since the last booking means higher risk, no training needed.
    baseline_score = X_test['recency_days']

    model_hits, k    = top_decile_capture(y_test, y_prob)
    baseline_hits, _ = top_decile_capture(y_test, baseline_score)

    metrics = {
        'cutoff'            : TRAIN_CUTOFF,
        'horizon_days'      : CHURN_HORIZON_DAYS,
        'n_customers'       : len(train_set),
        'churn_rate'        : float(y.mean()),
        'n_test'            : len(y_test),
        'test_churners'     : int(y_test.sum()),
        'model_auc'         : roc_auc_score(y_test, y_prob),
        'baseline_auc'      : roc_auc_score(y_test, baseline_score),
        'top10_size'        : k,
        'model_top10_hits'  : model_hits,
        'baseline_top10_hits': baseline_hits,
        'random_top10_hits' : k * float(y_test.mean()),
    }

    current['churn_probability'] = model.predict_proba(current[MODEL_FEATURES])[:, 1].round(3)
    df = rfm_df.merge(current[['customer_id', 'churn_probability']],
                      on='customer_id', how='left')

    importance = pd.DataFrame({
        'feature'   : MODEL_FEATURES,
        'importance': model.feature_importances_.round(3)
    }).sort_values('importance', ascending=False)

    return df, metrics, importance



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
    df, m, importance = train_churn_model(df)
    print(f"    Cutoff {m['cutoff']}, churn = no booking in next {m['horizon_days']} days")
    print(f"    Customers: {m['n_customers']:,}  churn rate: {m['churn_rate']*100:.1f}%  "
          f"test set: {m['n_test']:,} ({m['test_churners']:,} churners)")
    print(f"    Model AUC    : {m['model_auc']:.3f}  (1.0 = perfect, 0.5 = random)")
    print(f"    Baseline AUC : {m['baseline_auc']:.3f}  (recency only)")
    print(f"    Churners in top 10% ({m['top10_size']:,} customers): "
          f"model {m['model_top10_hits']:,} · baseline {m['baseline_top10_hits']:,} · "
          f"random ≈ {m['random_top10_hits']:.0f}")
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
    return df, summary, m, importance, roi


if __name__ == "__main__":
    run_all()