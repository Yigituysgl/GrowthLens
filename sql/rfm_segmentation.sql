SELECT 'customers'      AS table_name, COUNT(*) AS rows FROM customers
UNION ALL
SELECT 'bookings',       COUNT(*) FROM bookings
UNION ALL
SELECT 'experiments',    COUNT(*) FROM experiments
UNION ALL
SELECT 'marketing_spend',COUNT(*) FROM marketing_spend;


WITH base_metrics AS (
    SELECT
        c.customer_id,
        c.market,
        c.segment,
        JULIANDAY('2024-12-31') - JULIANDAY(MAX(b.booking_date))
            AS days_since_last,
        COUNT(b.booking_date)       AS frequency,
        ROUND(SUM(b.spend_eur), 2)  AS monetary
    FROM customers c
    LEFT JOIN bookings b ON c.customer_id = b.customer_id
    GROUP BY c.customer_id, c.market, c.segment
),

rfm_scores AS (
    SELECT
        customer_id, market, segment,
        days_since_last, frequency, monetary,
        5 - NTILE(4) OVER (ORDER BY days_since_last DESC) AS r_score,
        NTILE(4) OVER (ORDER BY frequency ASC)            AS f_score,
        NTILE(4) OVER (ORDER BY monetary ASC)             AS m_score
    FROM base_metrics
)

SELECT
    customer_id,
    market,
    segment,
    COALESCE(days_since_last, 999)  AS recency_days,
    COALESCE(frequency, 0)          AS total_bookings,
    COALESCE(monetary, 0)           AS total_spend_eur,
    r_score, f_score, m_score,
    (r_score + f_score + m_score)   AS rfm_score,
    CASE
        WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 'VIP'
        WHEN r_score >= 3 AND f_score >= 2                   THEN 'Loyal'
        WHEN r_score >= 3 AND f_score <  2                   THEN 'New Customer'
        WHEN r_score <  2 AND f_score >= 3                   THEN 'At Risk'
        WHEN r_score <  2 AND f_score <  2                   THEN 'Lost'
        ELSE 'Needs Attention'
    END AS rfm_segment
FROM rfm_scores
ORDER BY rfm_score DESC;