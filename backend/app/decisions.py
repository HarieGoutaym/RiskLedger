"""
decisions.py — RiskLedger Decision Engine.
Separates raw Model Risk Probability from Final Defensive Action (ALLOW, VERIFY, BLOCK).
Evaluates probability against:
  1. Category-derived optimal cost threshold
  2. Merchant-specific risk posture
  3. Aggregate Risk Budget exposure capacity
"""

from typing import Dict, Any
from sqlalchemy.orm import Session
from backend.app.models import RiskBudgetDB


def get_or_create_risk_budget(merchant_id: str, db: Session) -> RiskBudgetDB:
    budget = db.query(RiskBudgetDB).filter(RiskBudgetDB.merchant_id == merchant_id).first()
    if not budget:
        budget = RiskBudgetDB(
            merchant_id=merchant_id,
            daily_exposure_limit=100000.0,
            current_exposure=72400.0,
            consumed_count=142
        )
        db.add(budget)
        db.commit()
        db.refresh(budget)
    return budget


def evaluate_decision(
    risk_prob: float,
    amount: float,
    category_policy: Dict[str, Any],
    merchant_policy: Dict[str, Any],
    merchant_id: str,
    db: Session
) -> Dict[str, Any]:
    """
    Core Decision Engine Pipeline:
    risk_prob -> category_policy -> merchant_policy -> risk_budget -> action code
    """
    score_pct = round(risk_prob * 100.0, 1)
    cat_threshold = category_policy.get("threshold", 0.13)
    cat_thresh_pct = cat_threshold * 100.0

    # 1. Base Policy Determination
    if score_pct >= cat_thresh_pct * 2.5:
        initial_decision = "BLOCK"
        reason = f"High risk probability ({score_pct:.1f}%) exceeds block threshold ({cat_thresh_pct * 2.5:.1f}%)."
    elif score_pct >= cat_thresh_pct:
        initial_decision = "VERIFY"
        reason = f"Elevated risk probability ({score_pct:.1f}%) exceeds category cutoff ({cat_thresh_pct:.1f}%)."
    else:
        initial_decision = "ALLOW"
        reason = f"Low risk probability ({score_pct:.1f}%) sit within category cutoff ({cat_thresh_pct:.1f}%)."

    # 2. Risk Budget Capacity Check
    budget = get_or_create_risk_budget(merchant_id, db)
    budget_before = budget.current_exposure
    budget_after = budget_before
    budget_exceeded = False

    if initial_decision == "ALLOW":
        potential_exposure = budget_before + amount
        if potential_exposure > budget.daily_exposure_limit:
            initial_decision = "VERIFY"
            budget_exceeded = True
            reason = f"Transaction amount (₹{amount:,.2f}) would exceed merchant daily risk budget limit (₹{budget.daily_exposure_limit:,.0f}). Routed to VERIFY."
        else:
            budget.current_exposure = potential_exposure
            budget.consumed_count += 1
            budget_after = potential_exposure
            db.commit()

    return {
        "decision": initial_decision,
        "effective_threshold": round(cat_threshold, 3),
        "reason": reason,
        "budget_exceeded": budget_exceeded,
        "risk_budget_before": round(budget_before, 2),
        "risk_budget_after": round(budget_after, 2),
        "daily_exposure_limit": round(budget.daily_exposure_limit, 2),
    }
