"""
main.py — FastAPI Application Entry Point for RiskLedger API.
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Query, Path as FastPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func
from sqlalchemy.orm import Session
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score, confusion_matrix

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.database import engine, Base, get_db
from backend.app.models import TransactionDB, AuditLogDB, MerchantPolicyDB, CategoryPolicyDB
from backend.app.schemas import (
    TransactionInput, ScoringResponse, CategoryPolicyUpdate,
    MerchantPolicyUpdate, RiskBudgetUpdate, DecisionAnalysisInput
)
from backend.app.policies import get_category_policies, get_merchant_policy, update_category_policy
from backend.app.decisions import get_or_create_risk_budget, evaluate_decision
from backend.app.scoring import process_transaction_scoring
from backend.app.audit import get_audit_logs
from backend.app.analytics import compute_decision_analysis, FP_COST, pr_auc
from backend.model import get_model_service

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="RiskLedger — Adaptive Payment Risk Intelligence API",
    description="Enterprise payment risk decision engine providing real-time probability scoring, financial loss optimization, decision policies, and SHAP attributions.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_FILE = ROOT_DIR / "data" / "transactions.csv"


@app.on_event("startup")
async def startup_event():
    get_model_service()
    # Seed transactions database if empty
    db = next(get_db())
    if db.query(TransactionDB).count() == 0 and DATA_FILE.exists():
        print("[Startup] Seeding database with transactions dataset...")
        df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        total_len = len(df)
        train_idx = int(total_len * 0.70)
        val_idx = int(total_len * 0.80)

        records = []
        for i, r in df.iterrows():
            split_type = "train" if i < train_idx else ("val" if i < val_idx else "test")
            records.append(TransactionDB(
                transaction_id=str(r["transaction_id"]),
                timestamp=pd.to_datetime(r["timestamp"]),
                merchant_id=str(r["merchant_id"]),
                merchant_category=str(r["merchant_category"]),
                amount=float(r["amount"]),
                hour_of_day=int(r["hour_of_day"]),
                device_age_days=float(r["device_age_days"]),
                is_new_device=int(r["is_new_device"]),
                distance_from_usual_location_km=float(r["distance_from_usual_location_km"]),
                seconds_since_last_transaction=float(r["seconds_since_last_transaction"]),
                txn_count_last_hour=int(r["txn_count_last_hour"]),
                customer_avg_amount_30d=float(r["customer_avg_amount_30d"]),
                amount_to_avg_ratio=float(r["amount_to_avg_ratio"]),
                failed_attempts_last_hour=int(r["failed_attempts_last_hour"]),
                is_fraud=int(r["is_fraud"]),
                split_type=split_type
            ))
            if len(records) >= 2000:
                db.bulk_save_objects(records)
                db.commit()
                records = []
        if records:
            db.bulk_save_objects(records)
            db.commit()
        print(f"[Startup] Seeding complete. {db.query(TransactionDB).count():,} records stored.")


@app.get("/api/health")
async def health():
    return {"status": "ok", "engine": "RiskLedger Infrastructure", "version": "1.0.0"}


@app.get("/api/overview")
async def get_overview(db: Session = Depends(get_db)):
    """
    Calculates portfolio-wide overview metrics from actual database records.
    """
    total_txns = db.query(TransactionDB).count()
    if total_txns == 0:
        return {"error": "Dataset empty"}

    service = get_model_service()
    
    # Query test set records for held-out financial loss metrics
    test_records = db.query(TransactionDB).filter(TransactionDB.split_type == "test").all()
    if not test_records:
        test_records = db.query(TransactionDB).limit(10000).all()

    df_test = pd.DataFrame([{
        "amount": r.amount,
        "hour_of_day": r.hour_of_day,
        "device_age_days": r.device_age_days,
        "is_new_device": r.is_new_device,
        "distance_from_usual_location_km": r.distance_from_usual_location_km,
        "seconds_since_last_transaction": r.seconds_since_last_transaction,
        "txn_count_last_hour": r.txn_count_last_hour,
        "customer_avg_amount_30d": r.customer_avg_amount_30d,
        "amount_to_avg_ratio": r.amount_to_avg_ratio,
        "failed_attempts_last_hour": r.failed_attempts_last_hour,
        "is_fraud": r.is_fraud,
    } for r in test_records])

    probs = service.xgb_model.predict_proba(df_test[service.features])[:, 1]
    preds = (probs >= service.threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(df_test["is_fraud"], preds).ravel()
    fp_cost = float(fp * FP_COST)
    fn_cost = float(df_test["amount"].values[(preds == 0) & (df_test["is_fraud"].values == 1)].sum())
    total_cost = fp_cost + fn_cost

    actual_fraud_cnt = int(db.query(TransactionDB).filter(TransactionDB.is_fraud == 1).count())
    fraud_rate = round((actual_fraud_cnt / total_txns) * 100.0, 2)
    total_vol_sum = float(db.query(func.sum(TransactionDB.amount)).scalar() or 0.0)

    return {
        "transactions": total_txns,
        "high_risk": actual_fraud_cnt,
        "fraud_prevalence_pct": fraud_rate,
        "total_volume": round(total_vol_sum, 2),
        "fraud_exposure": round(fn_cost, 2),
        "false_positive_cost": round(fp_cost, 2),
        "false_negative_cost": round(fn_cost, 2),
        "total_modeled_loss": round(total_cost, 2),
        "operating_threshold": service.threshold,
    }


@app.get("/api/transactions")
async def get_transactions(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    search: Optional[str] = None,
    risk: Optional[str] = None,
    decision: Optional[str] = None,
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Paginated transactions API with server-side filtering.
    """
    query = db.query(TransactionDB).order_by(TransactionDB.timestamp.desc())

    if search:
        s = f"%{search}%"
        query = query.filter(
            (TransactionDB.transaction_id.like(s)) |
            (TransactionDB.merchant_id.like(s)) |
            (TransactionDB.merchant_category.like(s))
        )
    if category:
        query = query.filter(TransactionDB.merchant_category == category)

    total_count = query.count()
    offset = (page - 1) * limit
    records = query.offset(offset).limit(limit).all()

    service = get_model_service()
    res = []
    
    for r in records:
        df_single = pd.DataFrame([{
            f: getattr(r, f, 0.0) for f in service.features
        }])
        prob = float(service.xgb_model.predict_proba(df_single[service.features])[0, 1])
        score_100 = round(prob * 100.0, 1)

        cat_policy = service.category_policies.get(r.merchant_category, {"threshold": service.threshold})
        effective_thresh = float(cat_policy.get("threshold", service.threshold))
        thresh_pct = effective_thresh * 100.0

        if score_100 >= thresh_pct * 2.5:
            band = "HIGH"
            rec = "BLOCK"
        elif score_100 >= thresh_pct:
            band = "MEDIUM"
            rec = "VERIFY"
        else:
            band = "LOW"
            rec = "ALLOW"

        if risk and risk.upper() != "ALL" and band != risk.upper():
            continue
        if decision and decision.upper() != "ALL" and rec != decision.upper():
            continue

        res.append({
            "transaction_id": r.transaction_id,
            "timestamp": str(r.timestamp),
            "merchant_id": r.merchant_id,
            "merchant_category": r.merchant_category,
            "amount": r.amount,
            "risk_score": score_100,
            "risk_probability": round(prob, 4),
            "risk_band": band,
            "recommendation": rec,
            "effective_threshold": effective_thresh,
            "is_fraud_ground_truth": r.is_fraud,
        })

    return {
        "page": page,
        "limit": limit,
        "total": total_count,
        "transactions": res
    }


@app.get("/api/transactions/{transaction_id}")
async def get_transaction_by_id(transaction_id: str, db: Session = Depends(get_db)):
    r = db.query(TransactionDB).filter(TransactionDB.transaction_id == transaction_id).first()
    if not r:
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")

    service = get_model_service()
    payload = TransactionInput(
        transaction_id=r.transaction_id,
        amount=r.amount,
        customer_avg_amount_30d=r.customer_avg_amount_30d,
        failed_attempts_last_hour=r.failed_attempts_last_hour,
        hour_of_day=r.hour_of_day,
        distance_from_usual_location_km=r.distance_from_usual_location_km,
        merchant_category=r.merchant_category,
        merchant_id=r.merchant_id,
        device_age_days=r.device_age_days,
        is_new_device=r.is_new_device,
        seconds_since_last_transaction=r.seconds_since_last_transaction,
        txn_count_last_hour=r.txn_count_last_hour
    )
    return process_transaction_scoring(payload, service, db)


@app.post("/api/score", response_model=ScoringResponse)
async def score_transaction(payload: TransactionInput, db: Session = Depends(get_db)):
    try:
        service = get_model_service()
        return process_transaction_scoring(payload, service, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/policies")
async def get_policies(merchant_id: str = "merchant_a", db: Session = Depends(get_db)):
    return {
        "merchant_policy": get_merchant_policy(merchant_id, db),
        "category_policies": get_category_policies(db),
        "note": "Thresholds are derived from category-level financial-loss optimization on validation set and are not manually assigned."
    }


@app.patch("/api/policies/{category}")
async def patch_category_policy(category: str, update: CategoryPolicyUpdate, db: Session = Depends(get_db)):
    return update_category_policy(category, update.threshold, db)


@app.get("/api/risk-budget")
async def get_risk_budget(merchant_id: str = "merchant_a", db: Session = Depends(get_db)):
    budget = get_or_create_risk_budget(merchant_id, db)
    utilization_pct = round((budget.current_exposure / budget.daily_exposure_limit) * 100.0, 1)
    remaining = max(0.0, budget.daily_exposure_limit - budget.current_exposure)

    return {
        "merchant_id": budget.merchant_id,
        "daily_exposure_limit": round(budget.daily_exposure_limit, 2),
        "current_exposure": round(budget.current_exposure, 2),
        "remaining_capacity": round(remaining, 2),
        "utilization_pct": utilization_pct,
        "transactions_consuming_exposure": budget.consumed_count,
        "policy_rule": "The risk budget limits aggregate exposure rather than evaluating transactions only in isolation."
    }


@app.patch("/api/risk-budget")
async def update_risk_budget(update: RiskBudgetUpdate, merchant_id: str = "merchant_a", db: Session = Depends(get_db)):
    budget = get_or_create_risk_budget(merchant_id, db)
    budget.daily_exposure_limit = update.daily_exposure_limit
    db.commit()
    db.refresh(budget)
    return {"merchant_id": merchant_id, "daily_exposure_limit": budget.daily_exposure_limit}


@app.get("/api/threshold-analysis")
async def get_threshold_analysis(db: Session = Depends(get_db)):
    """
    Returns validation-locked threshold cost curve and optimal operating point.
    """
    service = get_model_service()
    val_records = db.query(TransactionDB).filter(TransactionDB.split_type == "val").all()
    if not val_records:
        val_records = db.query(TransactionDB).limit(5000).all()

    df_val = pd.DataFrame([{
        "amount": r.amount,
        "hour_of_day": r.hour_of_day,
        "device_age_days": r.device_age_days,
        "is_new_device": r.is_new_device,
        "distance_from_usual_location_km": r.distance_from_usual_location_km,
        "seconds_since_last_transaction": r.seconds_since_last_transaction,
        "txn_count_last_hour": r.txn_count_last_hour,
        "customer_avg_amount_30d": r.customer_avg_amount_30d,
        "amount_to_avg_ratio": r.amount_to_avg_ratio,
        "failed_attempts_last_hour": r.failed_attempts_last_hour,
        "is_fraud": r.is_fraud,
    } for r in val_records])

    probs = service.xgb_model.predict_proba(df_val[service.features])[:, 1]
    y_val = df_val["is_fraud"].values
    amts_val = df_val["amount"].values

    threshold_curve = []
    for t in np.linspace(0.02, 0.98, 49):
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        cost_fp = float(fp * FP_COST)
        cost_fn = float(amts_val[(preds == 0) & (y_val == 1)].sum())
        total_cost = cost_fp + cost_fn

        threshold_curve.append({
            "threshold": round(float(t), 3),
            "precision": round(float(precision_score(y_val, preds, zero_division=0)), 3),
            "recall": round(float(recall_score(y_val, preds, zero_division=0)), 3),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "cost_fp": round(cost_fp, 2),
            "cost_fn": round(cost_fn, 2),
            "total_cost": round(total_cost, 2)
        })

    return {
        "operating_threshold": service.threshold,
        "cost_curve": threshold_curve
    }


@app.post("/api/decision-analysis")
async def decision_analysis(payload: DecisionAnalysisInput):
    service = get_model_service()
    cat_policy = service.category_policies.get(payload.merchant_category, {"threshold": service.threshold})
    effective_thresh = float(cat_policy.get("threshold", service.threshold))
    return compute_decision_analysis(service.xgb_model, service.features, payload.model_dump(), effective_thresh)


@app.get("/api/audit-log")
async def get_audit_log_endpoint(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    decision: Optional[str] = None,
    db: Session = Depends(get_db)
):
    return get_audit_logs(db, limit=limit, offset=offset, decision=decision)


@app.get("/api/evaluate")
async def evaluate_model(db: Session = Depends(get_db)):
    """
    Returns actual held-out 10,000 test set metrics (3-way chronological split).
    Zero data leakage: threshold locked on validation set.
    """
    service = get_model_service()
    test_records = db.query(TransactionDB).filter(TransactionDB.split_type == "test").all()
    if not test_records:
        test_records = db.query(TransactionDB).limit(10000).all()

    df_test = pd.DataFrame([{
        "amount": r.amount,
        "hour_of_day": r.hour_of_day,
        "device_age_days": r.device_age_days,
        "is_new_device": r.is_new_device,
        "distance_from_usual_location_km": r.distance_from_usual_location_km,
        "seconds_since_last_transaction": r.seconds_since_last_transaction,
        "txn_count_last_hour": r.txn_count_last_hour,
        "customer_avg_amount_30d": r.customer_avg_amount_30d,
        "amount_to_avg_ratio": r.amount_to_avg_ratio,
        "failed_attempts_last_hour": r.failed_attempts_last_hour,
        "is_fraud": r.is_fraud,
    } for r in test_records])

    X_test, y_test = df_test[service.features], df_test["is_fraud"].values
    amts_test = df_test["amount"].values

    xgb_probs = service.xgb_model.predict_proba(X_test)[:, 1]
    logreg_probs = service.logreg.predict_proba(service.scaler.transform(X_test))[:, 1]

    p_xgb, r_xgb, _ = precision_recall_curve(y_test, xgb_probs)
    pr_auc_xgb = float(auc(r_xgb, p_xgb))

    p_lr, r_lr, _ = precision_recall_curve(y_test, logreg_probs)
    pr_auc_lr = float(auc(r_lr, p_lr))

    opt_preds = (xgb_probs >= service.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, opt_preds).ravel()
    opt_prec = float(precision_score(y_test, opt_preds, zero_division=0))
    opt_rec = float(recall_score(y_test, opt_preds, zero_division=0))
    opt_f1 = float(f1_score(y_test, opt_preds, zero_division=0))

    cost_fp = float(fp * FP_COST)
    cost_fn = float(amts_test[(opt_preds == 0) & (y_test == 1)].sum())
    total_cost = cost_fp + cost_fn

    return {
        "test_size": len(test_records),
        "fraud_cases": int(y_test.sum()),
        "pr_auc": round(pr_auc_xgb, 3),
        "pr_auc_baseline_logreg": round(pr_auc_lr, 3),
        "recall": round(opt_rec, 3),
        "precision": round(opt_prec, 3),
        "f1": round(opt_f1, 3),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "false_positive_cost": round(cost_fp, 2),
        "false_negative_cost": round(cost_fn, 2),
        "total_cost": round(total_cost, 2),
        "operating_threshold": service.threshold,
        "category_policies": service.category_policies
    }


# Mount static frontend
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse(status_code=404, content={"error": "Frontend index.html not found"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
