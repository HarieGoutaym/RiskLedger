"""
schemas.py — Pydantic Request & Response Validation Schemas for RiskLedger API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class TransactionInput(BaseModel):
    transaction_id: Optional[str] = "txn_preview"
    amount: float = Field(..., gt=0, example=48500.0)
    customer_avg_amount_30d: float = Field(..., gt=0, example=2500.0)
    failed_attempts_last_hour: int = Field(0, ge=0, example=2)
    hour_of_day: int = Field(..., ge=0, le=23, example=3)
    distance_from_usual_location_km: float = Field(..., ge=0, example=85.0)
    merchant_category: str = Field("electronics", example="electronics")
    merchant_id: Optional[str] = Field("merchant_a", example="merchant_a")
    device_age_days: Optional[float] = Field(1.5, ge=0)
    is_new_device: Optional[int] = Field(1, ge=0, le=1)
    seconds_since_last_transaction: Optional[float] = Field(45.0, ge=0)
    txn_count_last_hour: Optional[int] = Field(4, ge=0)
    payment_method: Optional[str] = Field("upi", example="upi")


class ShapAttribution(BaseModel):
    feature: str
    label: str
    val: float
    shap_value: float
    abs_shap: float
    impact: str
    explanation: str


class ScoringResponse(BaseModel):
    transaction_id: str
    risk_probability: float  # 0.0 to 1.0
    risk_score: float        # 0.0 to 100.0
    decision: str            # ALLOW, VERIFY, BLOCK
    effective_threshold: float
    merchant_category: str
    merchant_id: str
    explanation: List[ShapAttribution]
    model_version: str
    base_fraud_rate: float
    category_policy: Dict[str, Any]
    risk_budget_status: Dict[str, Any]
    counterfactual: Dict[str, Any]


class CategoryPolicyUpdate(BaseModel):
    threshold: float = Field(..., gt=0.0, lt=1.0)


class MerchantPolicyUpdate(BaseModel):
    allow_threshold: Optional[float] = None
    verify_threshold: Optional[float] = None
    block_threshold: Optional[float] = None


class RiskBudgetUpdate(BaseModel):
    daily_exposure_limit: float = Field(..., gt=0.0)


class DecisionAnalysisInput(BaseModel):
    amount: float = Field(..., gt=0)
    customer_avg_amount_30d: float = Field(2500.0, gt=0)
    failed_attempts_last_hour: int = Field(2, ge=0)
    hour_of_day: int = Field(3, ge=0, le=23)
    distance_from_usual_location_km: float = Field(85.0, ge=0)
    merchant_category: str = Field("electronics")
    device_age_days: float = Field(1.5, ge=0)
    is_new_device: int = Field(1, ge=0, le=1)
    seconds_since_last_transaction: float = Field(45.0, ge=0)
    txn_count_last_hour: int = Field(4, ge=0)
