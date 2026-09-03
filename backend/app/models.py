"""
models.py — SQLAlchemy ORM Data Models for RiskLedger.
Tables: transactions, category_policies, merchant_policies, risk_budgets, audit_logs, model_runs.
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, Text, ForeignKey, JSON
from datetime import datetime
from backend.app.database import Base


class TransactionDB(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(64), unique=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    merchant_id = Column(String(64), index=True, default="merchant_a")
    merchant_category = Column(String(64), index=True)
    amount = Column(Float, nullable=False)
    hour_of_day = Column(Integer)
    device_age_days = Column(Float)
    is_new_device = Column(Integer)
    distance_from_usual_location_km = Column(Float)
    seconds_since_last_transaction = Column(Float)
    txn_count_last_hour = Column(Integer)
    customer_avg_amount_30d = Column(Float)
    amount_to_avg_ratio = Column(Float)
    failed_attempts_last_hour = Column(Integer)
    is_fraud = Column(Integer, default=0)
    split_type = Column(String(16), default="train")  # train, val, test


class CategoryPolicyDB(Base):
    __tablename__ = "category_policies"

    category = Column(String(64), primary_key=True)
    threshold = Column(Float, nullable=False)
    fraud_rate = Column(Float, default=0.03)
    posture = Column(String(32), default="Moderate")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MerchantPolicyDB(Base):
    __tablename__ = "merchant_policies"

    merchant_id = Column(String(64), primary_key=True)
    allow_threshold = Column(Float, default=0.13)
    verify_threshold = Column(Float, default=0.30)
    block_threshold = Column(Float, default=0.30)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RiskBudgetDB(Base):
    __tablename__ = "risk_budgets"

    merchant_id = Column(String(64), primary_key=True)
    daily_exposure_limit = Column(Float, default=100000.0)
    current_exposure = Column(Float, default=72400.0)
    consumed_count = Column(Integer, default=142)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuditLogDB(Base):
    __tablename__ = "audit_logs"

    audit_id = Column(String(64), primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    transaction_id = Column(String(64), index=True)
    merchant_id = Column(String(64), index=True)
    merchant_category = Column(String(64))
    amount = Column(Float)
    model_version = Column(String(32), default="v1.0-xgb")
    risk_probability = Column(Float)
    risk_score = Column(Float)
    effective_threshold = Column(Float)
    decision = Column(String(32))  # ALLOW, VERIFY, BLOCK
    reason = Column(Text)
    shap_explanation = Column(JSON)
    risk_budget_before = Column(Float)
    risk_budget_after = Column(Float)


class ModelRunDB(Base):
    __tablename__ = "model_runs"

    run_id = Column(String(64), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    selected_model_type = Column(String(32))
    pr_auc_xgb = Column(Float)
    pr_auc_logreg = Column(Float)
    locked_threshold = Column(Float)
    test_recall = Column(Float)
    test_precision = Column(Float)
    test_f1 = Column(Float)
    test_total_cost = Column(Float)
