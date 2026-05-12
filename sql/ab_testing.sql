SELECT
    experiment,
    variant,
    COUNT(*)                                    AS total_users,
    SUM(converted)                              AS conversions,
    ROUND(AVG(converted) * 100, 4)              AS conversion_rate_pct,
    ROUND(AVG(ad_spend_eur), 4)                 AS avg_spend_per_user,
    ROUND(SUM(ad_spend_eur), 2)                 AS total_spend,
    ROUND(
        SUM(ad_spend_eur) / NULLIF(SUM(converted), 0)
    , 2)                                        AS cpa_eur
FROM experiments
GROUP BY experiment, variant
ORDER BY experiment, variant