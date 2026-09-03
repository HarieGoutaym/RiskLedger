"""
audit.py — Traceable Audit Ledger Logging & Query Service for RiskLedger.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from backend.app.models import AuditLogDB


def record_audit_event(
    db: Session,
    transaction_id: str,
    merchant_id: str,
    merchant_category: str,
    amount: float,
    model_version: str,
    risk_probability: float,
    effective_threshold: float,
    decision: str,
    reason: str,
    shap_explanation: List[Dict[str, Any]],
    risk_budget_before: float,
    risk_budget_after: float,
) -> AuditLogDB:
    audit_id = f"aud_{uuid.uuid4().hex[:12]}"
    log_entry = AuditLogDB(
        audit_id=audit_id,
        timestamp=datetime.utcnow(),
        transaction_id=transaction_id,
        merchant_id=merchant_id,
        merchant_category=merchant_category,
        amount=amount,
        model_version=model_version,
        risk_probability=risk_probability,
        risk_score=round(risk_probability * 100.0, 1),
        effective_threshold=effective_threshold,
        decision=decision,
        reason=reason,
        shap_explanation=shap_explanation,
        risk_budget_before=risk_budget_before,
        risk_budget_after=risk_budget_after,
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    return log_entry


def get_audit_logs(
    db: Session,
    limit: int = 50,
    offset: int = 0,
    decision: Optional[str] = None,
    merchant_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    query = db.query(AuditLogDB).order_by(AuditLogDB.timestamp.desc())
    if decision:
        query = query.filter(AuditLogDB.decision == decision.upper())
    if merchant_id:
        query = query.filter(AuditLogDB.merchant_id == merchant_id)

    records = query.offset(offset).limit(limit).all()
    return [
        {
            "audit_id": r.audit_id,
            "timestamp": str(r.timestamp),
            "transaction_id": r.transaction_id,
            "merchant_id": r.merchant_id,
            "merchant_category": r.merchant_category,
            "amount": r.amount,
            "model_version": r.model_version,
            "risk_probability": r.risk_probability,
            "risk_score": r.risk_score,
            "effective_threshold": r.effective_threshold,
            "decision": r.decision,
            "reason": r.reason,
            "shap_explanation": r.shap_explanation,
            "risk_budget_before": r.risk_budget_before,
            "risk_budget_after": r.risk_budget_after,
        } for r in records
    ]
