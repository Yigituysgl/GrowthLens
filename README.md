# 🔭 GrowthLens

**End-to-end growth analytics platform covering the full acquisition → retention → monetization loop.**

Built to demonstrate real-world data science skills: SQL analytics, machine learning, statistical testing, and interactive dashboards — all connected through a single synthetic travel marketplace dataset.


---

## The Business Problem

A travel marketplace (think GetYourGuide, Airbnb Experiences) needs to answer three questions every week:

1. **Who are our customers and which ones are at risk of leaving?** → CRM & Retention
2. **Did our last marketing experiment actually work?** → A/B Testing
3. **Are we pricing our experiences optimally?** → Pricing & Revenue

GrowthLens answers all three in one platform.

---

## What's Inside

###  CRM & Retention
- **RFM segmentation** of 50,000 customers using SQL window functions (`NTILE`, `JULIANDAY`, CTEs)
- **Churn prediction** with Random Forest — identifies customers most likely to stop booking
- **Interactive campaign simulator** — drag sliders to model win-back email ROI in real time
- Segments: VIP · Loyal · Needs Attention · At Risk · Lost

###  A/B Testing
- **Z-test engine** built from scratch using `scipy.stats` — no black-box libraries
- Tests two live experiments: urgency banner vs discount email campaign
- Outputs: Z-score, p-value, lift %, CPA comparison, and plain-English business recommendation
- Correctly identifies one experiment as **significant** and one as **not significant** — showing the difference between real signal and noise

###  Pricing & Revenue
- **Price elasticity modeling** from month-over-month booking and price data
- **Discount optimizer** — simulates revenue at every discount level (0–50%) to find the optimal point
- Revenue breakdown by market (Berlin, Paris, London, Amsterdam, Rome, Barcelona, Prague, Vienna) and category
- Channel ROI comparison: Paid Search vs CRM Email vs Organic

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Database | SQLite · DBeaver |
| SQL | CTEs · Window functions · JULIANDAY · NULLIF · COALESCE |
| Python | pandas · numpy · scikit-learn · scipy · plotly |
| Dashboard | Streamlit |
| ML | Random Forest (churn) · Price elasticity regression |
| Stats | Z-test for proportions · AUC evaluation |
| DevOps | Git · GitHub · Streamlit Cloud |

---

## Project Structure

```
GrowthLens/
├── app.py                        # Streamlit dashboard (4 pages)
├── requirements.txt
│
├── modules/
│   ├── data_generator.py         # Generates 50K customers + 95K bookings
│   ├── crm_retention.py          # RFM + churn model + campaign ROI
│   ├── ab_testing.py             # Z-test engine for A/B experiments
│   └── pricing_opt.py            # Elasticity model + discount optimizer
│
└── sql/
    ├── rfm_segmentation.sql      # CTE chain: base → scores → segments
    ├── ab_testing.sql            # Experiment aggregation by variant
    ├── pricing_analysis.sql      # Revenue by month/market/category
    └── marketing_roi.sql         # Channel spend vs revenue
```

---

## The Data

Fully synthetic dataset generated with `data_generator.py` — designed to reflect realistic travel marketplace patterns:

| Table | Rows | Description |
|-------|------|-------------|
| `customers` | 50,000 | Demographics, market, segment, email opt-in |
| `bookings` | 95,754 | Booking history across 8 markets, 10 categories, 3 channels |
| `experiments` | 18,000 | Two A/B tests with built-in conversion signals |
| `marketing_spend` | 576 | Monthly spend by market and channel |

**Why synthetic?** Full control over the business story — real-world signals (one significant experiment, one not; VIPs going quiet; elasticity below 1) are baked in intentionally.

---

## Key Findings

- **19,030 VIP customers** (38% of base) generate **64.7% of total revenue** — but their average recency is 159 days, signalling churn risk
- **Discount email campaign**: Z-score 4.91, p < 0.001 → statistically significant, 51% lift in conversions — but CPA increases from €4.63 to €6.68 (deploy selectively)
- **Urgency banner**: Z-score 1.738, p = 0.082 → not significant — do not ship, run longer
- **Price elasticity = 0.34** → customers are inelastic; deep discounting hurts revenue
- **6,541 customers** signed up but never booked → onboarding failure, not churn

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/Yigituysgl/GrowthLens.git
cd GrowthLens

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate the database
python modules/data_generator.py

# Launch the dashboard
streamlit run app.py
```

---

## About

Built by **Yigit Uysaloglu** — Data Scientist based in Berlin.

MSc in Data Analytics · Berlin School of Business and Innovation · 2026

Skills demonstrated: SQL analytics · Machine learning · Statistical testing · Python · Streamlit · Business storytelling

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Yigit_Uysaloglu-blue)](https://linkedin.com/in/yigit-uysaloglu)
[![GitHub](https://img.shields.io/badge/GitHub-Yigituysgl-black)](https://github.com/Yigituysgl)
