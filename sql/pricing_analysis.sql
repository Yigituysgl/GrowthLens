
SELECT
    STRFTIME('%Y-%m', b.booking_date)   AS month,
    b.market,
    b.category,
    b.channel,
    COUNT(*)                            AS total_bookings,
    ROUND(AVG(b.spend_eur), 2)          AS avg_price_eur,
    ROUND(SUM(b.spend_eur), 2)          AS total_revenue_eur
FROM bookings b
GROUP BY
    STRFTIME('%Y-%m', b.booking_date),
    b.market,
    b.category,
    b.channel
ORDER BY month, market