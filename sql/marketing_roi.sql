
SELECT
    ms.market,
    ms.channel,
    ROUND(SUM(ms.spend_eur), 2)         AS total_marketing_spend,
    COUNT(b.booking_date)               AS total_bookings,
    ROUND(SUM(b.spend_eur), 2)          AS total_revenue,
    ROUND(
        (SUM(b.spend_eur) - SUM(ms.spend_eur))
        / NULLIF(SUM(ms.spend_eur), 0) * 100
    , 1)                                AS roi_pct
FROM marketing_spend ms
LEFT JOIN bookings b
    ON ms.market  = b.market
    AND ms.channel = b.channel
GROUP BY ms.market, ms.channel
ORDER BY roi_pct DESC