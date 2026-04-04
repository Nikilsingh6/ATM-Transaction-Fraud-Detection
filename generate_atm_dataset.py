import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

DATASET_FILE = "synthetic_fraud_dataset.csv"
NUM_ROWS = 50000
RANDOM_SEED = 42

LOCATIONS = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad"]
CARD_TYPES = ["Visa", "Mastercard", "RuPay", "Amex"]
TRANSACTION_TYPES = [
    "ATM Withdrawal",
    "Cash Deposit",
    "Balance Inquiry",
    "Mini Statement",
]
AUTH_METHODS = ["PIN", "Biometric", "OTP", "Card + PIN"]
ATM_LOCATION_PROFILES = ["Usual Area", "Nearby Area", "Out of Area"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sample_transaction_amount(rng, transaction_type, suspicious):
    if transaction_type == "Balance Inquiry":
        return 0.0
    if transaction_type == "Mini Statement":
        return float(np.round(rng.uniform(0, 30), 2))
    if transaction_type == "Cash Deposit":
        amount = rng.gamma(shape=3.8 if suspicious else 2.8, scale=3200 if suspicious else 2200)
        return float(np.round(np.clip(amount, 100, 40000), 2))

    amount = rng.gamma(shape=4.2 if suspicious else 2.4, scale=3000 if suspicious else 1800)
    return float(np.round(np.clip(amount, 100, 35000), 2))


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    start_time = datetime(2025, 1, 1, 0, 0, 0)

    rows = []
    for i in range(1, NUM_ROWS + 1):
        suspicious = rng.random() < 0.24
        transaction_type = rng.choice(
            TRANSACTION_TYPES,
            p=[0.68, 0.12, 0.12, 0.08] if suspicious else [0.56, 0.18, 0.16, 0.10]
        )
        location = rng.choice(LOCATIONS)
        card_type = rng.choice(CARD_TYPES, p=[0.34, 0.28, 0.28, 0.10])
        authentication_method = rng.choice(
            AUTH_METHODS,
            p=[0.30, 0.08, 0.26, 0.36] if suspicious else [0.52, 0.16, 0.06, 0.26]
        )

        transaction_amount = sample_transaction_amount(rng, transaction_type, suspicious)
        account_balance = float(np.round(np.clip(rng.normal(85000, 42000), 500, 500000), 2))
        previous_fraudulent_activity = int(rng.random() < (0.27 if suspicious else 0.04))
        daily_transaction_count = int(rng.integers(5, 19) if suspicious else rng.integers(1, 9))
        failed_transaction_count_7d = int(rng.integers(2, 8) if suspicious else rng.integers(0, 3))
        transactions_last_1h = int(rng.integers(3, 9) if suspicious else rng.integers(0, 4))
        card_age = int(rng.integers(30, 480))
        transaction_distance = float(np.round(
            rng.uniform(25, 300) if suspicious else rng.uniform(0, 18),
            2
        ))
        if transaction_distance <= 10:
            atm_location_profile = "Usual Area"
        elif transaction_distance <= 60:
            atm_location_profile = "Nearby Area"
        else:
            atm_location_profile = "Out of Area"

        hour = int(rng.choice(
            [0, 1, 2, 3, 4, 5, 6, 22, 23] if suspicious else list(range(7, 22)),
            replace=True
        ))
        minute = int(rng.integers(0, 60))
        day_offset = int(rng.integers(0, 180))
        timestamp = start_time + timedelta(days=day_offset, hours=hour, minutes=minute)
        is_weekend = int(timestamp.weekday() >= 5)

        avg_transaction_amount_7d = float(np.round(np.clip(
            rng.normal(2200 if suspicious else 4800, 1800 if suspicious else 2200),
            0,
            28000
        ), 2))
        ip_address_flag = int(rng.random() < (0.18 if suspicious else 0.02))

        amount_ratio = transaction_amount / (account_balance + 1)
        late_night = int(hour <= 5 or hour >= 22)
        unusual_auth = int(authentication_method in {"OTP", "Card + PIN"} and transaction_type == "ATM Withdrawal")
        high_amount = int(transaction_amount >= 15000)
        very_high_distance = int(transaction_distance >= 120)
        velocity_flag = int(daily_transaction_count >= 10)
        burst_flag = int(transactions_last_1h >= 4)

        raw_risk = (
            -3.7
            + 3.8 * high_amount
            + 3.0 * very_high_distance
            + 2.4 * previous_fraudulent_activity
            + 2.0 * velocity_flag
            + 2.3 * burst_flag
            + 1.9 * late_night
            + 1.4 * unusual_auth
            + 1.6 * (failed_transaction_count_7d >= 3)
            + 1.0 * (amount_ratio >= 0.30)
            + 0.8 * is_weekend
            + 1.0 * (transaction_type == "ATM Withdrawal")
            + 1.6 * (atm_location_profile == "Out of Area")
            - 1.2 * (transaction_type == "Balance Inquiry")
            - 0.6 * (authentication_method == "Biometric")
            - 0.5 * (transaction_distance <= 3)
            + rng.normal(0, 0.75)
        )

        fraud_probability = sigmoid(raw_risk)
        fraud_label = int(rng.random() < fraud_probability)

        risk_score = np.clip(
            0.10
            + 0.55 * fraud_probability
            + 0.20 * high_amount
            + 0.10 * previous_fraudulent_activity
            + 0.08 * late_night
            + rng.normal(0, 0.06),
            0.01,
            0.99
        )

        rows.append({
            "Transaction_ID": f"ATM_TXN_{i:06d}",
            "User_ID": f"USER_{int(rng.integers(1000, 9999))}",
            "Transaction_Amount": transaction_amount,
            "Transaction_Type": transaction_type,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Account_Balance": account_balance,
            "Device_Type": "ATM Kiosk",
            "Location": location,
            "Merchant_Category": "ATM",
            "IP_Address_Flag": ip_address_flag,
            "Previous_Fraudulent_Activity": previous_fraudulent_activity,
            "Daily_Transaction_Count": daily_transaction_count,
            "Avg_Transaction_Amount_7d": avg_transaction_amount_7d,
            "Failed_Transaction_Count_7d": failed_transaction_count_7d,
            "Transactions_Last_1H": transactions_last_1h,
            "Card_Type": card_type,
            "Card_Age": card_age,
            "Transaction_Distance": transaction_distance,
            "ATM_Location_Profile": atm_location_profile,
            "Authentication_Method": authentication_method,
            "Risk_Score": float(np.round(risk_score, 4)),
            "Is_Weekend": is_weekend,
            "Fraud_Label": fraud_label,
        })

    df = pd.DataFrame(rows)
    df.to_csv(DATASET_FILE, index=False)

    print("Saved ATM-focused synthetic dataset:", DATASET_FILE)
    print("Shape:", df.shape)
    print("\nFraud label distribution:")
    print(df["Fraud_Label"].value_counts(normalize=True).round(4))
    print("\nTransaction type distribution:")
    print(df["Transaction_Type"].value_counts(normalize=True).round(4))
    print("\nAmount summary:")
    print(df["Transaction_Amount"].describe().round(2))


if __name__ == "__main__":
    main()
