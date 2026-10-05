import pandas as pd
import numpy as np
import sqlite3
import os
from scipy import stats

from data_generator import PLANTED_CONVERSION

DB_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "growthens.db")
SQL_PATH = os.path.join(os.path.dirname(__file__), "..", "sql", "ab_testing.sql")

ALPHA = 0.05
# A split this unlikely under 50/50 points to broken assignment, not chance.
SRM_THRESHOLD = 0.001


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



def lift_confidence_interval(conv_control, n_control, conv_test, n_test, alpha=ALPHA):
    p_c = conv_control / n_control
    p_t = conv_test / n_test
    z = stats.norm.ppf(1 - alpha / 2)

    diff = p_t - p_c
    se_diff = np.sqrt(p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_test)

    # The ratio of two rates is skewed, so its interval is built on the log scale.
    if conv_control > 0 and conv_test > 0:
        log_ratio = np.log(p_t / p_c)
        se_log = np.sqrt(1 / conv_test - 1 / n_test + 1 / conv_control - 1 / n_control)
        rel_low  = np.exp(log_ratio - z * se_log) - 1
        rel_high = np.exp(log_ratio + z * se_log) - 1
    else:
        rel_low = rel_high = float('nan')

    return {
        'diff'     : diff,
        'diff_low' : diff - z * se_diff,
        'diff_high': diff + z * se_diff,
        'rel_lift' : diff / p_c if p_c > 0 else float('nan'),
        'rel_low'  : rel_low,
        'rel_high' : rel_high,
    }



def power_two_proportions(p_control, p_test, n_control, n_test, alpha=ALPHA):
    z = stats.norm.ppf(1 - alpha / 2)
    p_bar = (n_control * p_control + n_test * p_test) / (n_control + n_test)
    # The test statistic uses the pooled SE (null), but its spread under the
    # real effect follows the unpooled SE.
    se_null = np.sqrt(p_bar * (1 - p_bar) * (1 / n_control + 1 / n_test))
    se_alt  = np.sqrt(p_control * (1 - p_control) / n_control
                      + p_test * (1 - p_test) / n_test)
    d = p_test - p_control
    return float(stats.norm.cdf((d - z * se_null) / se_alt)
                 + stats.norm.cdf((-d - z * se_null) / se_alt))



def sample_size_per_group(p_control, p_test, power=0.80, alpha=ALPHA):
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p_control + p_test) / 2
    n = ((z_a * np.sqrt(2 * p_bar * (1 - p_bar))
          + z_b * np.sqrt(p_control * (1 - p_control) + p_test * (1 - p_test))) ** 2
         / (p_test - p_control) ** 2)
    return int(np.ceil(n))



def sample_ratio_check(n_control, n_test, control_share=0.5, threshold=SRM_THRESHOLD):
    total = n_control + n_test
    expected = [total * control_share, total * (1 - control_share)]
    _, p_value = stats.chisquare([n_control, n_test], f_exp=expected)
    return {
        'control_share': n_control / total,
        'srm_p_value'  : float(p_value),
        'srm_ok'       : bool(p_value >= threshold),
    }



def compare_with_truth(experiment_name, n_control, n_test, ci, is_significant):
    truth = PLANTED_CONVERSION.get(experiment_name)
    if truth is None:
        return {}
    p_c, p_t = truth['control'], truth['test']
    true_lift = (p_t - p_c) / p_c
    has_effect = p_t != p_c
    return {
        'true_rate_control_pct': p_c * 100,
        'true_rate_test_pct'   : p_t * 100,
        'true_lift_pct'        : true_lift * 100,
        'ci_contains_truth'    : bool(ci['rel_low'] <= true_lift <= ci['rel_high']),
        'power_at_true_effect' : power_two_proportions(p_c, p_t, n_control, n_test),
        'n_needed_80'          : sample_size_per_group(p_c, p_t, 0.80),
        'n_needed_90'          : sample_size_per_group(p_c, p_t, 0.90),
        'verdict_correct'      : bool(is_significant == has_effect),
        'missed_real_effect'   : bool(has_effect and not is_significant),
    }



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
    is_significant = p_value < ALPHA
    cpa_control    = float(control['cpa_eur']) if control['cpa_eur'] else None
    cpa_test       = float(test['cpa_eur'])    if test['cpa_eur']    else None

    ci    = lift_confidence_interval(conv_control, n_control, conv_test, n_test)
    srm   = sample_ratio_check(n_control, n_test)
    truth = compare_with_truth(experiment_name, n_control, n_test, ci, is_significant)

    return {
        'diff_pp'          : round(ci['diff'] * 100, 3),
        'diff_low_pp'      : round(ci['diff_low'] * 100, 3),
        'diff_high_pp'     : round(ci['diff_high'] * 100, 3),
        'lift_low_pct'     : round(ci['rel_low'] * 100, 2),
        'lift_high_pct'    : round(ci['rel_high'] * 100, 2),
        **srm,
        **truth,
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
                                             lift_pct, cpa_control, cpa_test,
                                             truth.get('n_needed_80')),
    }



def explain(r):
    """Plain-language reading of one experiment result."""
    lines = []
    if not r['srm_ok']:
        lines.append(
            f"Warning: the split was {r['control_share']*100:.1f}% control instead of 50% "
            f"(p = {r['srm_p_value']:.2g}). Assignment looks broken, so do not trust this result."
        )
    lines.append(
        f"The test group converted at {r['rate_test_pct']:.2f}% vs {r['rate_control_pct']:.2f}% "
        f"for control, a lift of {r['lift_pct']:+.1f}%. The true lift is likely between "
        f"{r['lift_low_pct']:+.1f}% and {r['lift_high_pct']:+.1f}% (95% confidence interval)."
    )
    if r['lift_low_pct'] > 0:
        lines.append("The whole range is above zero, so the test variant really does convert better.")
    else:
        lines.append("The range includes zero, so this data cannot rule out that the change does nothing.")
    if r['srm_ok']:
        lines.append(f"Users were split {r['control_share']*100:.1f}% / "
                     f"{100 - r['control_share']*100:.1f}%, as planned.")
    if 'true_lift_pct' in r:
        lines.append(
            f"Because the data is synthetic, we know the answer: the generator planted "
            f"{r['true_rate_control_pct']:.1f}% → {r['true_rate_test_pct']:.1f}% "
            f"({r['true_lift_pct']:+.1f}%). With {r['n_control']:,} / {r['n_test']:,} users, "
            f"the test had a {r['power_at_true_effect']*100:.0f}% chance of detecting it; "
            f"80% power needs {r['n_needed_80']:,} users per group."
        )
        if r['missed_real_effect']:
            lines.append(
                "This test missed a real effect. It was underpowered, so 'not significant' "
                "here means 'not enough data', not 'no effect'."
            )
        elif r['verdict_correct']:
            lines.append("The verdict matches the planted truth.")
        if not r['ci_contains_truth']:
            lines.append("Note: the 95% interval does not contain the planted lift "
                         "(expected to happen in about 1 test out of 20).")
    return lines



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
                    cpa_control, cpa_test, n_needed_80=None):
    if not is_significant:
        if n_needed_80:
            return (f"Run the experiment longer: about {n_needed_80:,} users per group "
                    f"are needed for 80% power. Cannot distinguish signal from noise yet.")
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
        print(f"  Diff (pp)     : {r['diff_pp']:+.3f}  "
              f"95% CI [{r['diff_low_pp']:+.3f}, {r['diff_high_pp']:+.3f}]")
        print(f"  Lift 95% CI   : [{r['lift_low_pct']:+.2f}%, {r['lift_high_pct']:+.2f}%]")
        print(f"  Sample ratio  : {r['control_share']*100:.2f}% control, "
              f"p = {r['srm_p_value']:.3f} → {'OK' if r['srm_ok'] else 'MISMATCH'}")
        if 'true_lift_pct' in r:
            print(f"  Planted truth : {r['true_rate_control_pct']:.1f}% → "
                  f"{r['true_rate_test_pct']:.1f}% ({r['true_lift_pct']:+.1f}%), "
                  f"in CI: {r['ci_contains_truth']}")
            print(f"  Power         : {r['power_at_true_effect']*100:.1f}% at planted effect · "
                  f"needed for 80%: {r['n_needed_80']:,}/group · 90%: {r['n_needed_90']:,}/group")
            print(f"  Verdict right : {r['verdict_correct']}"
                  + ("  ← MISSED A REAL EFFECT (underpowered)" if r['missed_real_effect'] else ""))
        print(f"  Verdict       : {r['verdict']}")
        print(f"  Recommendation: {r['recommendation']}")
        print("  In plain words:")
        for line in explain(r):
            print(f"    - {line}")

    print("\n" + "=" * 55)
    print("Done! ab_testing.py is ready.\n")
    return results


if __name__ == "__main__":
    main()