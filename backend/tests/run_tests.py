"""
run_tests.py — Standalone Integration Test Execution Runner for RiskLedger Backend.
Tests live endpoints on http://127.0.0.1:8000.
"""

import urllib.request
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        return json.loads(resp.read().decode())

def post(path, data):
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL}{path}", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        return json.loads(resp.read().decode())

def patch(path, data):
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL}{path}", data=body, headers={"Content-Type": "application/json"}, method="PATCH")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        return json.loads(resp.read().decode())

def run():
    print("=== Running RiskLedger Backend Integration Test Suite ===")

    # 1. Health
    h = get("/api/health")
    assert h["status"] == "ok"
    print(" [PASS] GET /api/health")

    # 2. Overview
    ov = get("/api/overview")
    assert "transactions" in ov and "operating_threshold" in ov
    print(" [PASS] GET /api/overview")

    # 3. Scoring ALLOW
    payload_allow = {
        "amount": 1200.0,
        "customer_avg_amount_30d": 2500.0,
        "failed_attempts_last_hour": 0,
        "hour_of_day": 14,
        "distance_from_usual_location_km": 2.5,
        "merchant_category": "grocery",
        "merchant_id": "merchant_a"
    }
    sc_allow = post("/api/score", payload_allow)
    assert sc_allow["decision"] == "ALLOW"
    assert len(sc_allow["explanation"]) > 0
    print(" [PASS] POST /api/score (ALLOW decision)")

    # 4. Scoring BLOCK
    payload_block = {
        "amount": 85000.0,
        "customer_avg_amount_30d": 1200.0,
        "failed_attempts_last_hour": 5,
        "hour_of_day": 3,
        "distance_from_usual_location_km": 280.0,
        "merchant_category": "electronics",
        "merchant_id": "merchant_a"
    }
    sc_block = post("/api/score", payload_block)
    assert sc_block["decision"] in ["VERIFY", "BLOCK"]
    assert sc_block["risk_score"] > 50.0
    print(" [PASS] POST /api/score (BLOCK/VERIFY high-risk decision)")

    # 5. Policies
    pol = get("/api/policies?merchant_id=merchant_a")
    assert "category_policies" in pol
    patched = patch("/api/policies/electronics", {"threshold": 0.25})
    assert patched["threshold"] == 0.25
    print(" [PASS] GET/PATCH /api/policies")

    # 6. Risk Budget
    rb = get("/api/risk-budget?merchant_id=merchant_a")
    assert "current_exposure" in rb
    print(" [PASS] GET /api/risk-budget")

    # 7. Decision Analysis
    payload_da = {
        "amount": 48500.0,
        "customer_avg_amount_30d": 2500.0,
        "failed_attempts_last_hour": 2,
        "hour_of_day": 3,
        "distance_from_usual_location_km": 85.0,
        "merchant_category": "electronics"
    }
    da = post("/api/decision-analysis", payload_da)
    assert "boundaries" in da and len(da["curve"]) == 100
    print(" [PASS] POST /api/decision-analysis (100-point sweep)")

    # 8. Audit Log
    al = get("/api/audit-log?limit=10")
    assert isinstance(al, list)
    print(" [PASS] GET /api/audit-log")

    # 9. Evaluate
    ev = get("/api/evaluate")
    assert "pr_auc" in ev and "recall" in ev
    print(" [PASS] GET /api/evaluate")

    print("\nALL 9 INTEGRATION TEST SUITES PASSED SUCCESSFULLY (100%).")

if __name__ == "__main__":
    run()
