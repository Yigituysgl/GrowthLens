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

-- Only customers with at least one booking are ranked, so the large
-- never-booked group does not fill the bottom quartiles.
-- CUME_DIST gives tied values the same rank; NTILE would split ties arbitrarily.
ranked AS (
    SELECT
        customer_id, frequency,
        CUME_DIST() OVER (ORDER BY days_since_last DESC) AS r_pct,
        CUME_DIST() OVER (ORDER BY monetary ASC)         AS m_pct
    FROM base_metrics
    WHERE frequency > 0
),

rfm_scores AS (
    SELECT
        customer_id,
        -- The most recent bookers sit at the top of the distribution and get 4.
        CASE WHEN r_pct <= 0.25 THEN 1 WHEN r_pct <= 0.50 THEN 2
             WHEN r_pct <= 0.75 THEN 3 ELSE 4 END AS r_score,
        -- Booking counts are mostly 1, so quartiles cannot separate them;
        -- fixed count bands keep every score reachable and meaningful.
        CASE WHEN frequency = 1 THEN 1 WHEN frequency = 2 THEN 2
             WHEN frequency <= 4 THEN 3 ELSE 4 END AS f_score,
        CASE WHEN m_pct <= 0.25 THEN 1 WHEN m_pct <= 0.50 THEN 2
             WHEN m_pct <= 0.75 THEN 3 ELSE 4 END AS m_score
    FROM ranked
)

SELECT
    b.customer_id,
    b.market,
    b.segment,
    COALESCE(b.days_since_last, 999)  AS recency_days,
    b.frequency                       AS total_bookings,
    COALESCE(b.monetary, 0)           AS total_spend_eur,
    s.r_score, s.f_score, s.m_score,
    (s.r_score + s.f_score + s.m_score) AS rfm_score,
    CASE
        WHEN b.frequency = 0                                       THEN 'Never booked'
        WHEN s.r_score >= 3 AND s.f_score >= 3 AND s.m_score >= 3 THEN 'VIP'
        WHEN s.r_score >= 3 AND s.f_score >= 2                     THEN 'Loyal'
        WHEN s.r_score >= 3 AND s.f_score <  2                     THEN 'Recent one-time buyer'
        WHEN s.r_score <  2 AND s.f_score >= 3                     THEN 'At Risk'
        WHEN s.r_score <  2 AND s.f_score <  2                     THEN 'Lost'
        ELSE 'Needs Attention'
    END AS rfm_segment
FROM base_metrics b
LEFT JOIN rfm_scores s ON b.customer_id = s.customer_id
ORDER BY rfm_score DESC;
