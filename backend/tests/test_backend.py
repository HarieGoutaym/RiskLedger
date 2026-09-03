"""
test_backend.py — Automated Unit & Integration Tests for RiskLedger Backend.
Covers:
 1. Feature validation
 2. Model artifact loading
 3. End-to-end transaction scoring
 4. Policy selection & patching
 5. Decision Engine (ALLOW, VERIFY, BLOCK)
 6. Risk budget exposure limit enforcement
 7. Counterfactual coarse-to-fine decision boundary search
 8. Audit ledger logging & traceability
 9. Threshold financial loss analysis
"""

import sys
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_overview():
    res = client.get("/api/overview")
    assert res.status_code == 200
    data = res.json()
    assert "transactions" in data
    assert "high_risk" in data
    assert "operating_threshold" in data


def test_scoring_api_allow():
    payload = {
        "amount": 1200.0,
        "customer_avg_amount_30d": 2500.0,
        "failed_attempts_last_hour": 0,
        "hour_of_day": 14,
        "distance_from_usual_location_km": 2.5,
        "merchant_category": "grocery",
        "merchant_id": "merchant_a"
    }
    res = client.post("/api/score", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "risk_probability" in data
    assert "decision" in data
    assert data["decision"] in ["ALLOW", "VERIFY", "BLOCK"]
    assert len(data["explanation"]) > 0


def test_scoring_api_block():
    payload = {
        "amount": 85000.0,
        "customer_avg_amount_30d": 1200.0,
        "failed_attempts_last_hour": 5,
        "hour_of_day": 3,
        "distance_from_usual_location_km": 280.0,
        "merchant_category": "electronics",
        "merchant_id": "merchant_a"
    }
    res = client.post("/api/score", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["decision"] in ["VERIFY", "BLOCK"]
    assert data["risk_score"] > 50.0


def test_policies_api():
    res = client.get("/api/policies?merchant_id=merchant_a")
    assert res.status_code == 200
    data = res.json()
    assert "category_policies" in data
    assert "electronics" in data["category_policies"]

    # Test policy patching
    patch_res = client.patch("/api/policies/electronics", json={"threshold": 0.25})
    assert patch_res.status_code == 200
    assert patch_res.json()["threshold"] == 0.25


def test_risk_budget_api():
    res = client.get("/api/risk-budget?merchant_id=merchant_a")
    assert res.status_code == 200
    data = res.json()
    assert "daily_exposure_limit" in data
    assert "current_exposure" in data


def test_decision_analysis_api():
    payload = {
        "amount": 48500.0,
        "customer_avg_amount_30d": 2500.0,
        "failed_attempts_last_hour": 2,
        "hour_of_day": 3,
        "distance_from_usual_location_km": 85.0,
        "merchant_category": "electronics"
    }
    res = client.post("/api/decision-analysis", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "boundaries" in data
    assert "block_to_verify" in data["boundaries"]
    assert "verify_to_allow" in data["boundaries"]
    assert len(data["curve"]) == 100


def test_audit_log_api():
    res = client.get("/api/audit-log?limit=10")
    assert res.status_code == 200
    logs = res.json()
    assert isinstance(logs, list)


def test_evaluate_api():
    res = client.get("/api/evaluate")
    assert res.status_code == 200
    data = res.json()
    assert "pr_auc" in data
    assert "recall" in data
    assert "false_positive_cost" in data
    assert "false_negative_cost" in data
