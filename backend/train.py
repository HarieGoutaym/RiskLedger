"""
train.py — Trains, tunes, and evaluates PayGuard Risk Engine.

3-Way Chronological Split Architecture (Zero Data Leakage):
  1. Train Set (First 70% = 35,000 txns): Trains XGBoost & Logistic Regression.
  2. Validation Set (Middle 10% = 5,000 txns): Derives and LOCKS global & category-specific
     cost-minimizing decision thresholds C(t) = FP * 50 + sum(FN_Amount).
  3. Test Set (Final 20% = 10,000 txns / ~300 fraud cases): Pure held-out evaluation.
     NEVER USED FOR MODEL TUNING OR THRESHOLD SELECTION.
"""

import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, auc, precision_score, recall_score,
    f1_score, confusion_matrix,
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

FEATURES = [
    "amount", "hour_of_day", "device_age_days", "is_new_device",
    "distance_from_usual_location_km", "seconds_since_last_transaction",
    "txn_count_last_hour", "customer_avg_amount_30d", "amount_to_avg_ratio",
    "failed_attempts_last_hour",
]

FP_COST = 50            # friction cost of wrongly flagging a legitimate transaction
FN_COST_PER_RUPEE = 1.0  # a missed fraud costs the transaction amount


def pr_auc(y_true, probs):
    p, r, _ = precision_recall_curve(y_true, probs)
    return float(auc(r, p))


def main():
    df = pd.read_csv("data/transactions.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    total_len = len(df)
    train_idx = int(total_len * 0.70)  # 35,000 rows
    val_idx = int(total_len * 0.80)    # 5,000 rows

    train_df = df.iloc[:train_idx]
    val_df   = df.iloc[train_idx:val_idx]
    test_df  = df.iloc[val_idx:]

    X_train, y_train = train_df[FEATURES], train_df["is_fraud"]
    X_val, y_val     = val_df[FEATURES], val_df["is_fraud"]
    X_test, y_test   = test_df[FEATURES], test_df["is_fraud"]

    print(f"Train Set:      {len(train_df):,} rows ({y_train.sum():,} fraud cases)")
    print(f"Validation Set: {len(val_df):,} rows ({y_val.sum():,} fraud cases) [USED ONLY FOR THRESHOLD LOCKING]")
    print(f"Test Set:       {len(test_df):,} rows ({y_test.sum():,} fraud cases) [PURE HELD-OUT EVALUATION]\n")

    # --- 1. Train Baseline: Logistic Regression ---
    scaler = StandardScaler().fit(X_train)
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, solver="lbfgs")
    logreg.fit(scaler.transform(X_train), y_train)
    
    # --- 2. Train Main Model: XGBoost Classifier ---
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / max(pos, 1)

    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        min_child_weight=2,
        scale_pos_weight=scale_pos_weight * 0.5,
        eval_metric="aucpr",
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)

    # --- 3. Evaluate Models on Validation Set ---
    val_xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
    val_lr_probs  = logreg.predict_proba(scaler.transform(X_val))[:, 1]
    
    val_xgb_auc = pr_auc(y_val, val_xgb_probs)
    val_lr_auc  = pr_auc(y_val, val_lr_probs)

    print("=== Validation Performance Benchmark ===")
    print(f"Logistic Regression Val PR-AUC: {val_lr_auc:.3f}")
    print(f"XGBoost Classifier Val PR-AUC:  {val_xgb_auc:.3f}\n")

    selected_model_type = "xgb" if val_xgb_auc >= val_lr_auc else "logreg"

    # --- 4. Optimize and LOCK Operating Thresholds on Validation Set ONLY ---
    print("=== Threshold Optimization on Validation Set (No Test Set Leakage) ===")
    val_amounts = val_df["amount"].values
    val_y_arr   = y_val.values
    val_probs   = val_xgb_probs if selected_model_type == "xgb" else val_lr_probs

    best_threshold, best_val_cost = 0.5, float("inf")
    for t in np.linspace(0.02, 0.98, 97):
        preds = (val_probs >= t).astype(int)
        fp_mask = (preds == 1) & (val_y_arr == 0)
        fn_mask = (preds == 0) & (val_y_arr == 1)
        cost = fp_mask.sum() * FP_COST + (val_amounts[fn_mask] * FN_COST_PER_RUPEE).sum()
        if cost < best_val_cost:
            best_val_cost, best_threshold = cost, t

    print(f"LOCKED Global Cost-Minimizing Threshold: {best_threshold:.2f}")

    # Optimize and LOCK Category Thresholds on Validation Set
    category_policies = {}
    categories = val_df["merchant_category"].unique()

    for cat in sorted(categories):
        cat_mask = val_df["merchant_category"] == cat
        cat_y = val_y_arr[cat_mask]
        cat_probs = val_probs[cat_mask]
        cat_amts = val_amounts[cat_mask]

        if len(cat_y) == 0 or cat_y.sum() == 0:
            category_policies[cat] = {"threshold": float(best_threshold), "fraud_rate": 0.03}
            continue

        cat_best_t, cat_min_cost = float(best_threshold), float("inf")
        for t in np.linspace(0.02, 0.98, 97):
            c_preds = (cat_probs >= t).astype(int)
            c_fp = (c_preds == 1) & (cat_y == 0)
            c_fn = (c_preds == 0) & (cat_y == 1)
            c_cost = c_fp.sum() * FP_COST + (cat_amts[c_fn] * FN_COST_PER_RUPEE).sum()
            if c_cost < cat_min_cost:
                cat_min_cost, cat_best_t = c_cost, float(t)

        category_policies[cat] = {
            "threshold": round(cat_best_t, 3),
            "fraud_rate": round(float(cat_y.mean()), 4),
            "val_sample_count": int(len(cat_y)),
            "val_fraud_count": int(cat_y.sum()),
        }
        print(f"Category: {cat:<15} | LOCKED Validation Cutoff: {cat_best_t:.3f}")

    print("\n=== FINAL UNBIASED HELD-OUT TEST SET EVALUATION (10,000 Txns) ===")
    test_xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    test_lr_probs  = logreg.predict_proba(scaler.transform(X_test))[:, 1]
    
    test_xgb_auc = pr_auc(y_test, test_xgb_probs)
    test_lr_auc  = pr_auc(y_test, test_lr_probs)

    test_probs = test_xgb_probs if selected_model_type == "xgb" else test_lr_probs
    test_preds = (test_probs >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    amounts_test = test_df["amount"].values
    y_test_arr   = y_test.values

    fp_cost = fp * FP_COST
    fn_cost = amounts_test[(test_preds == 0) & (y_test_arr == 1)].sum()
    total_test_cost = fp_cost + fn_cost

    default_preds = (test_probs >= 0.5).astype(int)
    _, dfp, dfn, _ = confusion_matrix(y_test, default_preds).ravel()
    default_test_cost = dfp * FP_COST + amounts_test[(default_preds == 0) & (y_test_arr == 1)].sum()

    print(f"Held-Out Logistic Regression PR-AUC: {test_lr_auc:.3f}")
    print(f"Held-Out XGBoost PR-AUC:             {test_xgb_auc:.3f}")
    print(f"Selected Model Type:                {selected_model_type.upper()}")
    print(f"Locked Decision Threshold:          {best_threshold:.2f}")
    print(f"Test Precision:                     {precision_score(y_test, test_preds, zero_division=0):.3f}")
    print(f"Test Recall:                        {recall_score(y_test, test_preds, zero_division=0):.3f} ({tp:,} / {y_test.sum():,} fraud cases captured)")
    print(f"Test F1 Score:                      {f1_score(y_test, test_preds, zero_division=0):.3f}")
    print(f"Test Confusion Matrix -> TN: {tn:,} | FP: {fp:,} | FN: {fn:,} | TP: {tp:,}")
    print(f"False Positives Friction Cost (Rs.50/flag): Rs.{fp_cost:,.0f}")
    print(f"False Negatives Direct Fraud Loss:         Rs.{fn_cost:,.0f}")
    print(f"Total Expected Test Financial Loss:        Rs.{total_test_cost:,.0f}")
    print(f"(Default 0.5 Threshold Test Cost:          Rs.{default_test_cost:,.0f} | Net Savings: Rs.{default_test_cost - total_test_cost:,.0f})")

    # Save locked artifacts
    joblib.dump({
        "model": xgb_model,
        "scaler": scaler,
        "logreg": logreg,
        "selected_model_type": selected_model_type,
        "features": FEATURES,
        "threshold": float(best_threshold),
        "category_policies": category_policies,
        "pr_auc_xgb": float(test_xgb_auc),
        "pr_auc_logreg": float(test_lr_auc),
    }, "backend/xgb_model.joblib")

    print("\nSaved locked models & threshold policies to backend/xgb_model.joblib")


if __name__ == "__main__":
    main()
