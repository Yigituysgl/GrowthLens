-- Features use only bookings strictly before :cutoff, so the model never
-- sees anything from the period it is asked to predict.
WITH history AS (
    SELECT
        customer_id,
        JULIANDAY(:cutoff) - JULIANDAY(MAX(booking_date)) AS recency_days,
        COUNT(*)                                          AS frequency,
        ROUND(SUM(spend_eur), 2)                          AS total_spend_eur
    FROM bookings
    WHERE booking_date < :cutoff
    GROUP BY customer_id
),

-- Anyone who books inside [cutoff, label_end) is still active.
active_after AS (
    SELECT DISTINCT customer_id
    FROM bookings
    WHERE booking_date >= :cutoff
      AND booking_date <  :label_end
)

SELECT
    c.customer_id,
    c.market,
    h.recency_days,
    h.frequency,
    h.total_spend_eur,
    JULIANDAY(:cutoff) - JULIANDAY(c.signup_date) AS tenure_days,
    CASE WHEN a.customer_id IS NULL THEN 1 ELSE 0 END AS churned
FROM customers c
-- Inner join: customers with no booking before the cutoff have no history to learn from.
JOIN history h            ON c.customer_id = h.customer_id
LEFT JOIN active_after a  ON c.customer_id = a.customer_id
ORDER BY c.customer_id
