"""
policies.py — Category and Merchant Policy Management Engine for RiskLedger.
"""

from typing import Dict, Any
from sqlalchemy.orm import Session
from backend.app.models import CategoryPolicyDB, MerchantPolicyDB

DEFAULT_CATEGORY_POLICIES = {
    "digital_goods": {"threshold": 0.570, "fraud_rate": 0.0314, "posture": "Strict"},
    "electronics":   {"threshold": 0.140, "fraud_rate": 0.0348, "posture": "Moderate"},
    "fashion":       {"threshold": 0.130, "fraud_rate": 0.0281, "posture": "Moderate"},
    "grocery":       {"threshold": 0.460, "fraud_rate": 0.0319, "posture": "Lenient"},
    "travel":        {"threshold": 0.060, "fraud_rate": 0.0305, "posture": "Strict"},
    "utilities":     {"threshold": 0.220, "fraud_rate": 0.0346, "posture": "Moderate"},
}

DEFAULT_MERCHANT_POLICIES = {
    "merchant_a": {"allow_threshold": 0.13, "verify_threshold": 0.30, "block_threshold": 0.30},
    "merchant_b": {"allow_threshold": 0.15, "verify_threshold": 0.35, "block_threshold": 0.35},
    "merchant_c": {"allow_threshold": 0.10, "verify_threshold": 0.25, "block_threshold": 0.25},
}


def get_category_policies(db: Session) -> Dict[str, Any]:
    records = db.query(CategoryPolicyDB).all()
    if not records:
        # Seed default validation-locked policies
        for cat, info in DEFAULT_CATEGORY_POLICIES.items():
            db_obj = CategoryPolicyDB(
                category=cat,
                threshold=info["threshold"],
                fraud_rate=info["fraud_rate"],
                posture=info["posture"]
            )
            db.add(db_obj)
        db.commit()
        records = db.query(CategoryPolicyDB).all()

    return {
        r.category: {
            "threshold": r.threshold,
            "fraud_rate": r.fraud_rate,
            "posture": r.posture,
            "updated_at": str(r.updated_at)
        } for r in records
    }


def get_merchant_policy(merchant_id: str, db: Session) -> Dict[str, Any]:
    m = db.query(MerchantPolicyDB).filter(MerchantPolicyDB.merchant_id == merchant_id).first()
    if not m:
        defaults = DEFAULT_MERCHANT_POLICIES.get(merchant_id, {"allow_threshold": 0.13, "verify_threshold": 0.30, "block_threshold": 0.30})
        m = MerchantPolicyDB(merchant_id=merchant_id, **defaults)
        db.add(m)
        db.commit()
        db.refresh(m)

    return {
        "merchant_id": m.merchant_id,
        "allow_threshold": m.allow_threshold,
        "verify_threshold": m.verify_threshold,
        "block_threshold": m.block_threshold,
    }


def update_category_policy(category: str, new_threshold: float, db: Session) -> Dict[str, Any]:
    m = db.query(CategoryPolicyDB).filter(CategoryPolicyDB.category == category).first()
    if not m:
        m = CategoryPolicyDB(category=category, threshold=new_threshold, posture="Custom")
        db.add(m)
    else:
        m.threshold = new_threshold
    db.commit()
    db.refresh(m)
    return {"category": m.category, "threshold": m.threshold, "posture": m.posture}
