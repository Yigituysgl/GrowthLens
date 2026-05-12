import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta
import random


np.random.seed(42)
random.seed(42)


N_CUSTOMERS   = 50_000
N_MONTHS      = 24          # 2 years of history
START_DATE    = datetime(2023, 1, 1)

MARKETS = ["Berlin", "Paris", "London", "Amsterdam",
           "Rome", "Barcelona", "Prague", "Vienna"]

CATEGORIES = ["City Tours", "Museum Skip-Line", "Food & Wine",
              "Outdoor Adventures", "Day Trips", "Water Activities",
              "Cultural Shows", "Photography Tours",
              "Cooking Classes", "Nightlife Tours"]

CHANNELS = ["Paid Search", "CRM Email", "Organic"]

SEGMENTS = ["VIP", "Regular", "New", "At Risk"]

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "growthens.db")



def generate_customers():
    print("  Generating customers...")

    
    signup_offsets = np.random.randint(0, N_MONTHS * 30, N_CUSTOMERS)
    signup_dates   = [START_DATE + timedelta(days=int(d)) for d in signup_offsets]

    
    ages = np.random.normal(loc=38, scale=12, size=N_CUSTOMERS).clip(18, 75).astype(int)

    
    markets = np.random.choice(MARKETS, size=N_CUSTOMERS,
                               p=[0.20, 0.18, 0.17, 0.12, 0.12, 0.10, 0.06, 0.05])

    
    segments = np.random.choice(SEGMENTS, size=N_CUSTOMERS,
                                p=[0.10, 0.35, 0.30, 0.25])

    
    pref_categories = np.random.choice(CATEGORIES, size=N_CUSTOMERS)

    customers = pd.DataFrame({
        "customer_id"        : range(1, N_CUSTOMERS + 1),
        "signup_date"        : signup_dates,
        "age"                : ages,
        "market"             : markets,
        "segment"            : segments,
        "preferred_category" : pref_categories,
        "email_opt_in"       : np.random.choice([1, 0], size=N_CUSTOMERS, p=[0.75, 0.25]),
    })

    return customers



def generate_bookings(customers):
    print("  Generating bookings...")

    # How many bookings per year per segment
    bookings_per_year = {"VIP": 6.0, "Regular": 2.5, "New": 1.2, "At Risk": 0.4}

    # Average spend per booking per segment (€)
    avg_spend = {"VIP": 180, "Regular": 95, "New": 75, "At Risk": 60}

    rows = []
    for _, c in customers.iterrows():
        seg   = c["segment"]
        rate  = bookings_per_year[seg] / 12   # monthly rate
        base  = avg_spend[seg]

        # How many months since signup until end of dataset
        months_active = min(
            N_MONTHS,
            int((datetime(2024, 12, 31) - pd.Timestamp(c["signup_date"])).days / 30)
        )
        if months_active <= 0:
            continue

        for month_offset in range(months_active):
            # Poisson: realistic count of bookings this month
            n_bookings = np.random.poisson(rate)
            for _ in range(n_bookings):
                booking_date = pd.Timestamp(c["signup_date"]) + timedelta(
                    days=month_offset * 30 + np.random.randint(0, 30)
                )
                if booking_date > datetime(2024, 12, 31):
                    continue

                # Spend varies ±40% around base, always positive
                spend = max(15, np.random.normal(base, base * 0.4))

                rows.append({
                    "customer_id"   : c["customer_id"],
                    "booking_date"  : booking_date.date(),
                    "category"      : np.random.choice(
                        CATEGORIES,
                        p=([0.25] + [0.75/9] * 9)   # slight lean toward city tours
                    ),
                    "market"        : c["market"],
                    "channel"       : np.random.choice(
                        CHANNELS,
                        p=[0.50, 0.25, 0.25]         # paid search dominates
                    ),
                    "spend_eur"     : round(spend, 2),
                })

    bookings = pd.DataFrame(rows)
    print(f"    → {len(bookings):,} booking rows generated")
    return bookings



def generate_experiments(customers):
    print("  Generating A/B experiment data...")

    
    exp_a_customers = customers.sample(10_000, random_state=1)
    exp_a_customers = exp_a_customers.copy()
    exp_a_customers["experiment"]  = "urgency_banner"
    exp_a_customers["variant"]     = np.where(
        np.random.rand(len(exp_a_customers)) < 0.5, "control", "test"
    )
    # Control: 1.8% conversion | Test: 2.6% conversion (real signal)
    exp_a_customers["converted"] = exp_a_customers["variant"].apply(
        lambda v: int(np.random.rand() < (0.026 if v == "test" else 0.018))
    )
    exp_a_customers["ad_spend_eur"] = exp_a_customers["variant"].apply(
        lambda v: round(np.random.uniform(1.2, 2.8) if v == "test"
                        else np.random.uniform(0.9, 2.1), 2)
    )
    exp_a_customers["experiment_date"] = "2024-03-01"

    
    opted_in = customers[customers["email_opt_in"] == 1]
    exp_b_customers = opted_in.sample(min(8_000, len(opted_in)), random_state=2)
    exp_b_customers = exp_b_customers.copy()
    exp_b_customers["experiment"]  = "discount_email"
    exp_b_customers["variant"]     = np.where(
        np.random.rand(len(exp_b_customers)) < 0.5, "control", "test"
    )
    # Control: 12% open → 5% book | Test: 22% open → 8% book
    exp_b_customers["converted"] = exp_b_customers["variant"].apply(
        lambda v: int(np.random.rand() < (0.08 if v == "test" else 0.05))
    )
    exp_b_customers["ad_spend_eur"] = exp_b_customers["variant"].apply(
        lambda v: round(np.random.uniform(0.3, 0.8) if v == "test"
                        else np.random.uniform(0.1, 0.4), 2)
    )
    exp_b_customers["experiment_date"] = "2024-06-01"

    experiments = pd.concat([
        exp_a_customers[["customer_id","experiment","variant",
                         "converted","ad_spend_eur","experiment_date"]],
        exp_b_customers[["customer_id","experiment","variant",
                         "converted","ad_spend_eur","experiment_date"]]
    ]).reset_index(drop=True)

    print(f"    → {len(experiments):,} experiment rows generated")
    return experiments


def generate_marketing_spend():
    print("  Generating marketing spend data...")
    rows = []
    for month in range(N_MONTHS):
        period = (START_DATE + timedelta(days=month * 30)).strftime("%Y-%m")
        for market in MARKETS:
            for channel in CHANNELS:
                base = {"Paid Search": 15000, "CRM Email": 3000, "Organic": 500}[channel]
                # Berlin and London get bigger budgets
                market_mult = 1.4 if market in ["Berlin", "London"] else 1.0
                spend = base * market_mult * np.random.uniform(0.85, 1.15)
                rows.append({
                    "period"     : period,
                    "market"     : market,
                    "channel"    : channel,
                    "spend_eur"  : round(spend, 2),
                })
    return pd.DataFrame(rows)



def write_to_db(customers, bookings, experiments, marketing_spend):
    print(f"  Writing to database: {DB_PATH}")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    # if_exists="replace" drops and recreates the table each run
    # so you always have a clean fresh dataset
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    bookings.to_sql("bookings",   conn, if_exists="replace", index=False)
    experiments.to_sql("experiments", conn, if_exists="replace", index=False)
    marketing_spend.to_sql("marketing_spend", conn, if_exists="replace", index=False)

    conn.close()
    print("  Database written successfully.")



def generate_all():
    print("\nGrowthLens — generating synthetic dataset...")
    print("=" * 50)

    customers        = generate_customers()
    bookings         = generate_bookings(customers)
    experiments      = generate_experiments(customers)
    marketing_spend  = generate_marketing_spend()

    write_to_db(customers, bookings, experiments, marketing_spend)

    print("=" * 50)
    print(f"Done! Summary:")
    print(f"  Customers       : {len(customers):>8,}")
    print(f"  Bookings        : {len(bookings):>8,}")
    print(f"  Experiment rows : {len(experiments):>8,}")
    print(f"  Marketing rows  : {len(marketing_spend):>8,}")
    print(f"  Database        : {DB_PATH}")
    print()


if __name__ == "__main__":
    generate_all()