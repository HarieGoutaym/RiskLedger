"""
scoring.py — Core Scoring Pipeline Orchestrator for RiskLedger.
Pipeline: validate input -> feature engineering -> XGBoost prediction -> SHAP explanation
          -> category/merchant policy -> check risk budget -> decision -> audit record -> return result.
"""

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from backend.app.schemas import TransactionInput, ScoringResponse
from backend.app.policies import get_category_policies, get_merchant_policy
from backend.app.decisions import evaluate_decision
from backend.app.audit import record_audit_event
from backend.app.analytics import compute_decision_analysis


def process_transaction_scoring(
    payload: TransactionInput,
    model_service,
    db: Session
) -> ScoringResponse:
    # 1. Feature Engineering
    data = payload.model_dump()
    avg = max(data["customer_avg_amount_30d"], 1.0)
    if data.get("amount_to_avg_ratio") is None:
        data["amount_to_avg_ratio"] = round(data["amount"] / avg, 3)

    row = {f: data.get(f, 0.0) for f in model_service.features}
    df_single = pd.DataFrame([row])

    # 2. Model Prediction
    prob = float(model_service.xgb_model.predict_proba(df_single[model_service.features])[0, 1])
    score_100 = round(prob * 100.0, 1)

    # 3. SHAP Explanation
    explanation = model_service.explainer.explain(df_single, top_k=4)

    # 4. Policy Retrieval
    cat_policies = get_category_policies(db)
    cat = payload.merchant_category
    cat_info = cat_policies.get(cat, {"threshold": model_service.threshold, "fraud_rate": 0.03, "posture": "Moderate"})

    merchant_id = payload.merchant_id or "merchant_a"
    merchant_info = get_merchant_policy(merchant_id, db)

    # 5. Decision Engine & Risk Budget Check
    decision_res = evaluate_decision(
        risk_prob=prob,
        amount=payload.amount,
        category_policy=cat_info,
        merchant_policy=merchant_info,
        merchant_id=merchant_id,
        db=db
    )

    # 6. Counterfactual Linear Sweep
    cf_res = compute_decision_analysis(
        xgb_model=model_service.xgb_model,
        features=model_service.features,
        payload_dict=data,
        effective_threshold=decision_res["effective_threshold"]
    )

    # 7. Record Traceable Audit Event
    txn_id = payload.transaction_id or f"txn_preview"
    record_audit_event(
        db=db,
        transaction_id=txn_id,
        merchant_id=merchant_id,
        merchant_category=cat,
        amount=payload.amount,
        model_version="v1.0-xgb",
        risk_probability=prob,
        effective_threshold=decision_res["effective_threshold"],
        decision=decision_res["decision"],
        reason=decision_res["reason"],
        shap_explanation=explanation["top_reasons"],
        risk_budget_before=decision_res["risk_budget_before"],
        risk_budget_after=decision_res["risk_budget_after"]
    )

    return ScoringResponse(
        transaction_id=txn_id,
        risk_probability=round(prob, 4),
        risk_score=score_100,
        decision=decision_res["decision"],
        effective_threshold=decision_res["effective_threshold"],
        merchant_category=cat,
        merchant_id=merchant_id,
        explanation=explanation["top_reasons"],
        model_version="v1.0-xgb",
        base_fraud_rate=explanation["base_value_prob"],
        category_policy=cat_info,
        risk_budget_status={
            "before": decision_res["risk_budget_before"],
            "after": decision_res["risk_budget_after"],
            "limit": decision_res["daily_exposure_limit"],
            "budget_exceeded": decision_res["budget_exceeded"]
        },
        counterfactual=cf_res["boundaries"]
    )
