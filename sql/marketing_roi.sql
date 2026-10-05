-- Spend and revenue are each summed to one row per market and channel
-- before joining; joining the raw rows would repeat every booking once
-- per spend month and every spend month once per booking.
WITH spend AS (
    SELECT
        market,
        channel,
        SUM(spend_eur) AS total_marketing_spend
    FROM marketing_spend
    GROUP BY market, channel
),

revenue AS (
    SELECT
        market,
        channel,
        COUNT(*)       AS total_bookings,
        SUM(spend_eur) AS total_revenue
    FROM bookings
    GROUP BY market, channel
)

SELECT
    s.market,
    s.channel,
    ROUND(s.total_marketing_spend, 2)       AS total_marketing_spend,
    COALESCE(r.total_bookings, 0)           AS total_bookings,
    ROUND(COALESCE(r.total_revenue, 0), 2)  AS total_revenue,
    -- NULL when there is no spend: ROI is undefined, not infinitely good.
    ROUND(
        (COALESCE(r.total_revenue, 0) - s.total_marketing_spend)
        / NULLIF(s.total_marketing_spend, 0) * 100
    , 1)                                    AS roi_pct
FROM spend s
LEFT JOIN revenue r
    ON  s.market  = r.market
    AND s.channel = r.channel
ORDER BY roi_pct DESC
