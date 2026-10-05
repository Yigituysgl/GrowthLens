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
    opt_in    = pd.read_sql("SELECT customer_id, email_opt_in FROM customers", conn)
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
    df = (rfm_df
          .merge(current[['customer_id', 'churn_probability']], on='customer_id', how='left')
          .merge(opt_in, on='customer_id', how='left'))

    # Held-out customers were not used in training, so their observed re-booking
    # after the cutoff is an honest "no campaign" baseline for the simulator.
    holdout = pd.DataFrame({
        'customer_id'      : train_set.loc[X_test.index, 'customer_id'].values,
        'churn_probability': y_prob,
        'churned'          : y_test.values,
    }).merge(opt_in, on='customer_id', how='left')

    importance = pd.DataFrame({
        'feature'   : MODEL_FEATURES,
        'importance': model.feature_importances_.round(3)
    }).sort_values('importance', ascending=False)

    return df, metrics, importance, holdout



def _top_share_by_risk(df, target_pct):
    pool = df[(df['email_opt_in'] == 1) & df['churn_probability'].notna()]
    pool = pool.sort_values('churn_probability', ascending=False, kind='stable')
    return pool.head(int(round(len(pool) * target_pct)))


def select_campaign_targets(df, target_pct):
    # Only opted-in customers can be emailed; never-booked customers have no score.
    return _top_share_by_risk(df, target_pct)


def baseline_rebook_rate(holdout, target_pct):
    # Same selection rule on held-out customers, then read off who actually booked again.
    group = _top_share_by_risk(holdout, target_pct)
    return float(1 - group['churned'].mean()) if len(group) else 0.0



def campaign_economics(n_targeted, baseline_rate, uplift, avg_booking_value,
                       margin, cost_per_email, discount=0.0):
    would_book_anyway = n_targeted * baseline_rate
    # The campaign can only win back customers who would not have booked anyway.
    extra_rebookings  = min(n_targeted * uplift, n_targeted - would_book_anyway)

    extra_revenue = extra_rebookings * avg_booking_value * (1 - discount)
    # A discount comes straight off the price, so it cuts profit one-for-one.
    extra_profit  = extra_rebookings * avg_booking_value * (margin - discount)

    email_cost           = n_targeted * cost_per_email
    anyway_discount_cost = would_book_anyway * avg_booking_value * discount
    total_cost           = email_cost + anyway_discount_cost
    net_profit           = extra_profit - total_cost
    roi_pct = net_profit / total_cost * 100 if total_cost > 0 else float('nan')

    return {
        'n_targeted'          : n_targeted,
        'would_book_anyway'   : would_book_anyway,
        'extra_rebookings'    : extra_rebookings,
        'extra_revenue'       : extra_revenue,
        'extra_profit'        : extra_profit,
        'email_cost'          : email_cost,
        'anyway_discount_cost': anyway_discount_cost,
        'total_cost'          : total_cost,
        'net_profit'          : net_profit,
        'roi_pct'             : roi_pct,
    }



def simulate_campaign_roi(df, holdout, target_pct=0.20, uplift=0.02, margin=0.20,
                          cost_per_email=0.05, discount=0.0):
    targets = select_campaign_targets(df, target_pct)
    base    = baseline_rebook_rate(holdout, target_pct)
    avg_value = (targets['total_spend_eur'].sum() / targets['total_bookings'].sum()
                 if len(targets) else 0.0)

    result = campaign_economics(len(targets), base, uplift, avg_value,
                                margin, cost_per_email, discount)
    result.update({
        'n_eligible'       : len(select_campaign_targets(df, 1.0)),
        'baseline_rate'    : base,
        'avg_booking_value': avg_value,
    })
    return result



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
    df, m, importance, holdout = train_churn_model(df)
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

    print("\n[4] Campaign ROI simulation (defaults: top 20%, uplift 2 pp, "
          "margin 20%, €0.05/email, no discount):")
    roi = simulate_campaign_roi(df, holdout)
    print(f"    Eligible (opted in, scored): {roi['n_eligible']:,}")
    print(f"    Customers targeted   : {roi['n_targeted']:,}")
    print(f"    Avg booking value    : €{roi['avg_booking_value']:,.2f}  (from data)")
    print(f"    Re-book w/o campaign : {roi['baseline_rate']*100:.1f}%  "
          f"→ {roi['would_book_anyway']:,.0f} customers (not credited)")
    print(f"    Extra re-bookings    : {roi['extra_rebookings']:,.0f}")
    print(f"    Extra revenue        : €{roi['extra_revenue']:,.2f}")
    print(f"    Extra profit         : €{roi['extra_profit']:,.2f}")
    print(f"    Email cost           : €{roi['email_cost']:,.2f}")
    print(f"    Net profit           : €{roi['net_profit']:,.2f}")
    print(f"    ROI on profit        : {roi['roi_pct']:.1f}%")

    print("\n" + "=" * 50)
    print("Done! crm_retention.py is ready.")
    return df, summary, m, importance, roi


if __name__ == "__main__":
    run_all()