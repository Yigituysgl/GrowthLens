import pandas as pd
import numpy as np
import sqlite3
import os
from scipy import stats

DB_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "growthens.db")
SQL_PATH = os.path.join(os.path.dirname(__file__), "..", "sql", "ab_testing.sql")


def get_connection():
    return sqlite3.connect(DB_PATH)



def load_experiment_data():
    conn = get_connection()
    sql  = open(SQL_PATH).read()
    df   = pd.read_sql(sql, conn)
    conn.close()
    return df


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Z-test (manual implementation using scipy.stats.norm)
#
# Formula:
#   p_pool = (conv_A + conv_B) / (n_A + n_B)   ← pooled proportion
#   SE     = sqrt(p_pool * (1-p_pool) * (1/n_A + 1/n_B))
#   Z      = (rate_A - rate_B) / SE
#   p      = 2 * (1 - Φ(|Z|))                  ← two-tailed p-value
#
# Why two-tailed? We test for ANY difference (better OR worse).
# One-tailed would only test "is test better?" — riskier.
# ═══════════════════════════════════════════════════════════════════
def _ztest_proportions(conv_control, n_control, conv_test, n_test):
    rate_control = conv_control / n_control
    rate_test    = conv_test    / n_test

    # Pooled proportion — best estimate of true rate under null hypothesis
    p_pool = (conv_control + conv_test) / (n_control + n_test)

    # Standard error of the difference
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_test))

    # Z-score: how many standard deviations apart are the rates?
    z_score = (rate_test - rate_control) / se

    # Two-tailed p-value using normal distribution
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return z_score, p_value



def run_ztest(df, experiment_name):
    exp     = df[df['experiment'] == experiment_name]
    control = exp[exp['variant'] == 'control'].iloc[0]
    test    = exp[exp['variant'] == 'test'].iloc[0]

    n_control    = int(control['total_users'])
    n_test       = int(test['total_users'])
    conv_control = int(control['conversions'])
    conv_test    = int(test['conversions'])
    rate_control = conv_control / n_control
    rate_test    = conv_test    / n_test

    z_score, p_value = _ztest_proportions(
        conv_control, n_control, conv_test, n_test
    )

    lift_pct       = (rate_test - rate_control) / rate_control * 100
    is_significant = p_value < 0.05
    cpa_control    = float(control['cpa_eur']) if control['cpa_eur'] else None
    cpa_test       = float(test['cpa_eur'])    if test['cpa_eur']    else None

    return {
        'experiment'       : experiment_name,
        'n_control'        : n_control,
        'conv_control'     : conv_control,
        'rate_control_pct' : round(rate_control * 100, 3),
        'spend_control'    : round(float(control['total_spend']), 2),
        'cpa_control'      : round(cpa_control, 2) if cpa_control else None,
        'n_test'           : n_test,
        'conv_test'        : conv_test,
        'rate_test_pct'    : round(rate_test * 100, 3),
        'spend_test'       : round(float(test['total_spend']), 2),
        'cpa_test'         : round(cpa_test, 2) if cpa_test else None,
        'z_score'          : round(float(z_score), 3),
        'p_value'          : round(float(p_value), 6),
        'lift_pct'         : round(lift_pct, 2),
        'is_significant'   : is_significant,
        'verdict'          : _verdict(is_significant, lift_pct,
                                      cpa_control, cpa_test),
        'recommendation'   : _recommendation(experiment_name, is_significant,
                                             lift_pct, cpa_control, cpa_test),
    }



def _verdict(is_significant, lift_pct, cpa_control, cpa_test):
    if not is_significant:
        return "No significant difference — do not act on this result"
    if lift_pct > 0 and cpa_test and cpa_control and cpa_test < cpa_control:
        return "Test WINS — higher conversion AND lower cost per acquisition"
    if lift_pct > 0 and cpa_test and cpa_control and cpa_test >= cpa_control:
        return "Test converts better but costs more — deploy selectively"
    if lift_pct < 0:
        return "Control WINS — test underperforms, revert"
    return "Mixed result — review by segment before deciding"


def _recommendation(experiment_name, is_significant, lift_pct,
                    cpa_control, cpa_test):
    if not is_significant:
        return ("Run the experiment longer or increase sample size. "
                "Cannot distinguish signal from noise yet.")
    if 'urgency' in experiment_name:
        if lift_pct > 0:
            return (f"Roll out urgency banner to all markets. "
                    f"Expected {lift_pct:.1f}% uplift in conversion. "
                    f"Deploy on high-traffic days first and monitor CPA weekly.")
        return "Keep original banner. Urgency messaging is hurting conversion."
    if 'discount' in experiment_name:
        if lift_pct > 0:
            return (f"Discount email drives {lift_pct:.1f}% more bookings. "
                    f"Roll out to full opted-in base. "
                    f"Watch margin impact — consider tiered discount by segment.")
        return "Discount did not improve conversion. Try personalised offer instead."
    return "Significant result — consult product team before full rollout."



def run_all_experiments():
    df          = load_experiment_data()
    experiments = df['experiment'].unique()
    results     = [run_ztest(df, exp) for exp in experiments]
    return results, df



def main():
    print("\nGrowthLens — A/B Testing Analysis")
    print("=" * 55)

    results, _ = run_all_experiments()

    for r in results:
        print(f"\n── Experiment: {r['experiment']} ──────────────────")
        print(f"  {'Group':<10} {'Users':>8} {'Conversions':>12} "
              f"{'Conv. Rate':>12} {'CPA':>8}")
        print(f"  {'Control':<10} {r['n_control']:>8,} "
              f"{r['conv_control']:>12,} "
              f"{r['rate_control_pct']:>11.3f}% "
              f"€{r['cpa_control']:>7}")
        print(f"  {'Test':<10} {r['n_test']:>8,} "
              f"{r['conv_test']:>12,} "
              f"{r['rate_test_pct']:>11.3f}% "
              f"€{r['cpa_test']:>7}")
        print(f"  {'─'*53}")
        print(f"  Lift          : {r['lift_pct']:+.2f}%")
        print(f"  Z-score       : {r['z_score']}  "
              f"(threshold: ±1.96)")
        print(f"  P-value       : {r['p_value']}  "
              f"(threshold: 0.05)")
        print(f"  Significant   : {'✓ YES' if r['is_significant'] else '✗ NO'}")
        print(f"  Verdict       : {r['verdict']}")
        print(f"  Recommendation: {r['recommendation']}")

    print("\n" + "=" * 55)
    print("Done! ab_testing.py is ready.\n")
    return results


if __name__ == "__main__":
    main()