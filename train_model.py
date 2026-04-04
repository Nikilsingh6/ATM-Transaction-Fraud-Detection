import json
import pickle
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_FILE = "synthetic_fraud_dataset.csv"
MODEL_FILE = "model.pkl"
PREPROCESSOR_FILE = "preprocessor.pkl"
CONFIG_FILE = "model_config.json"


def detect_risk_score_scale(df):
    risk_min = float(df["Risk_Score"].min())
    risk_max = float(df["Risk_Score"].max())

    print("Risk_Score min:", risk_min)
    print("Risk_Score max:", risk_max)

    if 0 <= risk_min and risk_max <= 1:
        return "0_to_1"
    return "0_to_100"


def print_results(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred) * 100
    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)
    print("Accuracy:", round(accuracy, 2), "%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    return accuracy


def find_best_threshold(y_true, probabilities):
    candidate_thresholds = np.arange(0.3, 0.71, 0.05)
    best_threshold = 0.5
    best_f1 = -1.0
    best_recall = -1.0

    for threshold in candidate_thresholds:
        preds = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)

        # Bias threshold selection toward catching fraud cases instead of being overly conservative.
        if recall >= 0.60 and score > best_f1:
            best_f1 = score
            best_recall = recall
            best_threshold = float(round(threshold, 2))

    if best_f1 >= 0:
        return best_threshold, best_f1, best_recall

    for threshold in candidate_thresholds:
        preds = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_recall = recall
            best_threshold = float(round(threshold, 2))

    return best_threshold, best_f1, best_recall


def build_input_caps(df):
    return {
        "Transaction_Amount": float(df["Transaction_Amount"].quantile(0.99)),
        "Transaction_Distance": float(df["Transaction_Distance"].quantile(0.99)),
        "Failed_Transaction_Count_7d": float(df["Failed_Transaction_Count_7d"].quantile(0.99)),
        "Transactions_Last_1H": float(df["Transactions_Last_1H"].quantile(0.99)),
        "Risk_Score": float(df["Risk_Score"].max()),
    }


def main():
    # 1. Load dataset
    df = pd.read_csv(DATASET_FILE)
    print("Original shape:", df.shape)

    # 2. Clean data
    df = df.drop_duplicates()
    df = df.dropna()
    print("After cleaning:", df.shape)

    # 3. Detect risk score scale
    risk_score_scale = detect_risk_score_scale(df)

    if risk_score_scale == "0_to_1":
        high_risk_cutoff = 0.7
    else:
        high_risk_cutoff = 70.0

    print("Detected Risk Score scale:", risk_score_scale)
    print("High Risk cutoff:", high_risk_cutoff)
    input_caps = build_input_caps(df)

    # 4. Feature engineering
    df["Amount_Log"] = np.log1p(df["Transaction_Amount"])
    df["Distance_Risk"] = df["Transaction_Distance"] * df["Risk_Score"]

    # 5. Features and target
    feature_columns = [
        "Transaction_Amount",
        "Transaction_Type",
        "Transaction_Distance",
        "Failed_Transaction_Count_7d",
        "Transactions_Last_1H",
        "ATM_Location_Profile",
        "Authentication_Method",
        "Risk_Score",
        "Amount_Log",
        "Distance_Risk"
    ]

    target_column = "Fraud_Label"

    X = df[feature_columns]
    y = df[target_column]

    # 6. Numeric and categorical features
    numeric_features = [
        "Transaction_Amount",
        "Transaction_Distance",
        "Failed_Transaction_Count_7d",
        "Transactions_Last_1H",
        "Risk_Score",
        "Amount_Log",
        "Distance_Risk"
    ]

    categorical_features = [
        "Transaction_Type",
        "ATM_Location_Profile",
        "Authentication_Method"
    ]

    # 7. Train-validation-test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )

    # 8. Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()

    if hasattr(X_val_processed, "toarray"):
        X_val_processed = X_val_processed.toarray()

    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    # 9. Handle imbalance
    print("\nBefore SMOTE:")
    print(y_train.value_counts())

    smote = SMOTE(sampling_strategy=0.9, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    print("\nAfter SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    # 10. Train Logistic Regression model for ATM fraud detection
    logistic_model = LogisticRegression(
        max_iter=8000,
        solver="liblinear",
        C=2.0,
        random_state=42
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=logistic_model,
        method="sigmoid",
        cv=3
    )

    calibrated_model.fit(X_train_resampled, y_train_resampled)

    validation_probs = calibrated_model.predict_proba(X_val_processed)[:, 1]
    logistic_threshold, validation_f1, validation_recall = find_best_threshold(y_val, validation_probs)
    print("\nSelected fraud threshold from validation set:", logistic_threshold)
    print("Validation F1 score:", round(validation_f1, 4))
    print("Validation recall:", round(validation_recall, 4))

    logistic_probs = calibrated_model.predict_proba(X_test_processed)[:, 1]
    logistic_preds = (logistic_probs >= logistic_threshold).astype(int)

    logistic_accuracy = print_results(
        "ATM Fraud Detection - Logistic Regression Results",
        y_test,
        logistic_preds
    )

    with open(PREPROCESSOR_FILE, "wb") as f:
        pickle.dump(preprocessor, f)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(calibrated_model, f)

    config = {
        "risk_score_scale": risk_score_scale,
        "high_risk_cutoff": high_risk_cutoff,
        "prediction_threshold": logistic_threshold,
        "saved_model_name": "Calibrated Logistic Regression",
        "model_type": "calibrated_logistic_regression",
        "logistic_accuracy_percent": round(logistic_accuracy, 2),
        "validation_f1": round(validation_f1, 4),
        "validation_recall": round(validation_recall, 4),
        "best_accuracy_percent": round(logistic_accuracy, 2),
        "input_caps": input_caps,
        "dashboard_input_fields": [
            "Transaction Amount",
            "Transaction Type",
            "Failed PIN Attempts",
            "Transactions in Last 1 Hour",
            "ATM Location Profile",
            "Auth Method"
        ],
        "feature_columns": feature_columns
    }

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print("\n" + "=" * 50)
    print("FINAL SAVED MODEL")
    print("=" * 50)
    print("Saved Model: Logistic Regression")
    print("Accuracy:", round(logistic_accuracy, 2), "%")
    print("Files saved successfully")


if __name__ == "__main__":
    main()
