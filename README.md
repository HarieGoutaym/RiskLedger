# RiskLedger — Adaptive Payment Risk Intelligence
> **Razorpay Buildathon · Track 02 (AI Risk Manager)**

> *"RiskLedger is a cost-aware payment risk engine that doesn't use an arbitrary 0.5 fraud threshold. It learns fraud probability, determines the financially optimal operating point, adapts that decision to merchant risk policies, and explains every prediction using model-derived SHAP values."*

---

## Architecture & Zero-Data-Leakage Methodology

RiskLedger uses a **3-Way Chronological Split** across 50,000 synthetic transactions:
1. **Train Set (First 70% = 35,000 txns)**: Trains XGBoost Classifier & Logistic Regression Baseline.
2. **Validation Set (Middle 10% = 5,000 txns)**: Derives and **LOCKS** global cost-minimizing cutoff (`t* = 0.13`) and category-specific thresholds (`C(t) = False Positives × ₹50 + Direct Fraud Loss`).
3. **Held-Out Test Set (Final 20% = 10,000 txns / 319 fraud cases)**: **NEVER USED FOR MODEL TUNING OR THRESHOLD SELECTION**. Evaluated purely out-of-sample for unbiased metrics.

---

## Out-Of-Sample Performance (10,000 Held-Out Test Transactions)

| Parameter / Metric | Baseline (Logistic Regression) | Selected Main Model (XGBoost) |
|---------------------|-------------------------------|-------------------------------|
| **Held-Out PR-AUC** | **0.885** | **0.888** (Selected via metric) |
| **Validation-Locked Cutoff (t*)** | N/A | **0.13** (Locked on Val Set) |
| **Test Precision @ t*** | N/A | **21.3%** |
| **Test Recall @ t*** | N/A | **96.6%** (**308 of 319 test fraud cases captured**) |
| **Friction Cost (False Positives @ ₹50)** | N/A | ₹56,950 (1,139 false flags) |
| **Direct Fraud Loss (False Negatives)** | N/A | ₹49,492 (11 missed fraud cases) |
| **Total Out-of-Sample Financial Loss** | N/A | **₹106,442** |
| **Savings vs Default 0.5 Cutoff** | N/A | **₹67,535 saved (38.8% cost reduction)** |

> **Defensive Mandate:** RiskLedger does not initiate payments, move funds, or optimize merchant revenue. It strictly provides defensive risk scoring and policy assessment.

---

## Core System Capabilities

### 1. Metric-Driven Model Selection & SHAP TreeExplainer
- Tuned XGBoost (`max_depth=3`, `n_estimators=400`, `learning_rate=0.03`, `subsample=0.8`, `scale_pos_weight` calibrated).
- XGBoost achieved **PR-AUC 0.888**, outperforming Logistic Regression (`0.885`). Using `shap.TreeExplainer` on the winning XGBoost model is 100% statistically valid.

### 2. Validation-Locked Category Threshold Policies
- Category cutoffs are **not manually assigned**; they are derived on the validation set:
  - `digital_goods`: Cutoff **0.570**
  - `electronics`: Cutoff **0.140**
  - `fashion`: Cutoff **0.130**
  - `grocery`: Cutoff **0.460**
  - `travel`: Cutoff **0.060**
  - `utilities`: Cutoff **0.220**

### 3. Coarse-to-Fine Sweep Counterfactual Search
- Sweeps transaction amounts across 100 candidate evaluation points to find the smallest amount change that crosses the category policy threshold without non-monotonicity assumptions.

---

## Execution Commands

```powershell
cd (The repo folder)

# Start Production Server:
.\.venv\Scripts\python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```

Access the dashboard at **`http://localhost:8000`**.
