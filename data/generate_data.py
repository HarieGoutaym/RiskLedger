"""
Synthetic transaction generator for PayGuard Risk Engine.

Design choices, on purpose:
- 50,000 transactions (spans 120 days across 3 merchant accounts and 6 categories).
- Realistic 3% fraud rate (~1,500 total fraud cases; ~300 in 10,000 held-out test set).
- Non-trivial feature overlap (legitimate late-night txns, travel spend, new devices)
  forcing non-linear learning without artificial class separability.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)
N = 50000
DAYS = 120
MERCHANTS = ["merchant_a", "merchant_b", "merchant_c"]
CATEGORIES = ["electronics", "grocery", "travel", "fashion", "digital_goods", "utilities"]

start = datetime(2026, 1, 1)


def _hour_weights(odd_bias):
    odd_hours = list(range(0, 5)) + [23]
    weights = np.ones(24)
    total_odd_slots = len(odd_hours)
    total_day_slots = 24 - total_odd_slots
    weights[odd_hours] = odd_bias / total_odd_slots
    day_hours = [h for h in range(24) if h not in odd_hours]
    weights[day_hours] = (1 - odd_bias) / total_day_slots
    return weights / weights.sum()

offsets_minutes = np.sort(RNG.uniform(0, DAYS * 24 * 60, size=N))

FRAUD_RATE = 0.03

def main():
    rows = []
    for i in range(N):
        ts = start + timedelta(minutes=float(offsets_minutes[i]))
        merchant = RNG.choice(MERCHANTS)
        category = RNG.choice(CATEGORIES)

        is_fraud = int(RNG.random() < FRAUD_RATE)

        # Baseline amount vary by category slightly for realistic cost curves
        cat_base_mult = {
            "digital_goods": 1.4,
            "electronics": 1.6,
            "travel": 1.8,
            "fashion": 1.0,
            "grocery": 0.6,
            "utilities": 0.8
        }.get(category, 1.0)

        customer_avg = max(200.0, RNG.normal(3000 * cat_base_mult, 1200))

        if is_fraud:
            ratio = max(0.1, RNG.lognormal(mean=0.9, sigma=0.85))
            is_new_device = int(RNG.random() < 0.55)
            hour = int(RNG.choice(range(24), p=_hour_weights(odd_bias=0.45)))
            distance = max(0.0, RNG.exponential(38))
            txn_velocity = int(RNG.poisson(2.3))
            failed_attempts = int(RNG.poisson(0.55))
        else:
            ratio = max(0.1, RNG.lognormal(mean=-0.05, sigma=0.45))
            is_new_device = int(RNG.random() < 0.06)
            hour = int(RNG.choice(range(24), p=_hour_weights(odd_bias=0.08)))
            distance = max(0.0, RNG.exponential(9))
            txn_velocity = int(RNG.poisson(0.8))
            failed_attempts = int(RNG.poisson(0.12))

        amount = round(customer_avg * ratio, 2)
        device_age = max(0.0, RNG.exponential(15 if is_new_device else 220))
        seconds_since_last = max(1.0, RNG.exponential(600 if txn_velocity > 2 else 3600))

        rows.append(dict(
            transaction_id=f"txn_{i:06d}",
            timestamp=ts,
            merchant_id=merchant,
            merchant_category=category,
            amount=round(amount, 2),
            hour_of_day=hour,
            device_age_days=round(device_age, 1),
            is_new_device=is_new_device,
            distance_from_usual_location_km=round(distance, 1),
            seconds_since_last_transaction=round(seconds_since_last, 1),
            txn_count_last_hour=txn_velocity,
            customer_avg_amount_30d=round(customer_avg, 2),
            amount_to_avg_ratio=round(ratio, 3),
            failed_attempts_last_hour=failed_attempts,
            is_fraud=is_fraud,
        ))

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df.to_csv("data/transactions.csv", index=False)

    print(f"Generated {len(df)} synthetic transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.3%} ({df['is_fraud'].sum()} fraud cases)")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

if __name__ == "__main__":
    main()
