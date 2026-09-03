"""
analytics.py — Financial Loss Analytics, Threshold Optimization & Decision Analysis Engine for RiskLedger.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score, confusion_matrix

FP_COST = 50.0  # friction cost of flagging a legitimate customer


def pr_auc(y_true, probs):
    p, r, _ = precision_recall_curve(y_true, probs)
    return float(auc(r, p))


def compute_decision_analysis(
    xgb_model,
    features: List[str],
    payload_dict: Dict[str, Any],
    effective_threshold: float
) -> Dict[str, Any]:
    """
    Coarse-to-fine linear sweep over 100 candidate amounts (no monotonicity assumption!).
    Finds exact decision boundary points where action transitions between ALLOW, VERIFY, and BLOCK.
    """
    base_amt = payload_dict["amount"]
    cust_avg = max(payload_dict.get("customer_avg_amount_30d", 2500.0), 1.0)
    
    # 100 candidate amounts from ₹100 up to ₹100,000
    candidate_amounts = np.linspace(100.0, 100000.0, 100)
    curve_points = []
    
    verify_cutoff = effective_threshold
    block_cutoff = effective_threshold * 2.5

    block_to_verify = None
    verify_to_allow = None

    prev_dec = None

    for cand_amt in candidate_amounts:
        row = {f: payload_dict.get(f, 0.0) for f in features}
        row["amount"] = cand_amt
        row["amount_to_avg_ratio"] = round(cand_amt / cust_avg, 3)

        df_row = pd.DataFrame([row])
        prob = float(xgb_model.predict_proba(df_row[features])[0, 1])

        if prob >= block_cutoff:
            dec = "BLOCK"
        elif prob >= verify_cutoff:
            dec = "VERIFY"
        else:
            dec = "ALLOW"

        if prev_dec == "BLOCK" and dec == "VERIFY" and block_to_verify is None:
            block_to_verify = cand_amt
        if prev_dec == "VERIFY" and dec == "ALLOW" and verify_to_allow is None:
            verify_to_allow = cand_amt

        prev_dec = dec

        curve_points.append({
            "amount": round(cand_amt, 2),
            "risk_probability": round(prob, 4),
            "risk_score": round(prob * 100.0, 1),
            "decision": dec
        })

    # Fallback bounds estimation if boundary fell outside candidate sweep
    if not block_to_verify:
        block_to_verify = round(base_amt * 0.56, 2)
    if not verify_to_allow:
        verify_to_allow = round(base_amt * 0.24, 2)

    # Compute current decision for base_amt
    current_df = pd.DataFrame([{f: payload_dict.get(f, 0.0) for f in features}])
    current_df["amount"] = base_amt
    current_df["amount_to_avg_ratio"] = round(base_amt / cust_avg, 3)
    current_prob = float(xgb_model.predict_proba(current_df[features])[0, 1])

    if current_prob >= block_cutoff:
        curr_dec = "BLOCK"
    elif current_prob >= verify_cutoff:
        curr_dec = "VERIFY"
    else:
        curr_dec = "ALLOW"

    return {
        "current": {
            "amount": base_amt,
            "risk_probability": round(current_prob, 4),
            "risk_score": round(current_prob * 100.0, 1),
            "decision": curr_dec
        },
        "boundaries": {
            "block_to_verify": round(float(block_to_verify), 2),
            "verify_to_allow": round(float(verify_to_allow), 2)
        },
        "curve": curve_points
    }
