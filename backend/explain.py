"""
explain.py — SHAP-based feature attribution engine for RiskLedger.

Uses shap.TreeExplainer on XGBoost to generate exact feature attributions
and analytical risk factors for enterprise fraud monitoring.
"""

import shap
import pandas as pd
import numpy as np

# Enterprise risk feature labels and analytical explanation templates
FEATURE_TEMPLATES = {
    "amount_to_avg_ratio": {
        "label": "Baseline Amount Ratio",
        "pos": lambda v: f"High amount ratio: {v:.1f}x of customer 30-day baseline",
        "neg": lambda v: f"Standard amount ratio: {v:.1f}x of customer baseline",
    },
    "amount": {
        "label": "Transaction Amount",
        "pos": lambda v: f"High transaction value (₹{v:,.2f})",
        "neg": lambda v: f"Low transaction value (₹{v:,.2f})",
    },
    "failed_attempts_last_hour": {
        "label": "Recent Failed Attempts",
        "pos": lambda v: f"Elevated failed attempts: {int(v)} prior decline(s) in last hour",
        "neg": lambda v: "No prior failed attempts in last hour",
    },
    "is_new_device": {
        "label": "Device Signature",
        "pos": lambda v: "Unrecognized device signature (new device ID)",
        "neg": lambda v: "Recognized customer device profile",
    },
    "device_age_days": {
        "label": "Device Tenure",
        "pos": lambda v: f"Short device tenure ({v:.1f} days since first seen)",
        "neg": lambda v: f"Established device profile ({v:.0f} days active tenure)",
    },
    "hour_of_day": {
        "label": "Transaction Time Window",
        "pos": lambda v: f"Off-peak transaction hour ({int(v):02d}:00 UTC)",
        "neg": lambda v: f"Standard operating window ({int(v):02d}:00 UTC)",
    },
    "distance_from_usual_location_km": {
        "label": "Geographic Deviation",
        "pos": lambda v: f"Geographic delta: {v:.1f} km from historical location centroid",
        "neg": lambda v: f"Geographic proximity: {v:.1f} km from historical centroid",
    },
    "txn_count_last_hour": {
        "label": "Short-Term Velocity",
        "pos": lambda v: f"High velocity: {int(v)} transactions in 1-hour window",
        "neg": lambda v: f"Normal velocity: {int(v)} transactions in 1-hour window",
    },
    "seconds_since_last_transaction": {
        "label": "Inter-Transaction Interval",
        "pos": lambda v: f"Rapid interval: {v:.0f} seconds since preceding transaction",
        "neg": lambda v: f"Standard interval: {v:.0f} seconds since preceding transaction",
    },
    "customer_avg_amount_30d": {
        "label": "30-Day Customer Baseline",
        "pos": lambda v: f"High baseline spending tier (₹{v:,.2f})",
        "neg": lambda v: f"Standard baseline spending tier (₹{v:,.2f})",
    },
}

class RiskExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain(self, df_single: pd.DataFrame, top_k: int = 4):
        """
        Computes exact SHAP feature attributions for a single transaction.
        """
        X = df_single[self.feature_names]
        shap_vals = self.explainer.shap_values(X)
        
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif len(shap_vals.shape) == 2:
            vals = shap_vals[0]
        else:
            vals = shap_vals

        base_val = self.explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])
        else:
            base_val = float(base_val)

        base_prob = round(1.0 / (1.0 + np.exp(-base_val)) * 100.0, 1)

        attributions = []
        for feat_name, shap_v in zip(self.feature_names, vals):
            raw_val = float(X[feat_name].iloc[0])
            shap_v = float(shap_v)
            
            templates = FEATURE_TEMPLATES.get(feat_name, {
                "label": feat_name,
                "pos": lambda v: f"{feat_name} = {v:.2f} (risk increase)",
                "neg": lambda v: f"{feat_name} = {v:.2f} (risk decrease)"
            })
            
            impact = "INCREASES_RISK" if shap_v >= 0 else "DECREASES_RISK"
            explanation_fn = templates["pos"] if shap_v >= 0 else templates["neg"]
            
            attributions.append({
                "feature": feat_name,
                "label": templates["label"],
                "val": round(raw_val, 2),
                "shap_value": round(shap_v, 4),
                "abs_shap": abs(shap_v),
                "impact": impact,
                "explanation": explanation_fn(raw_val),
            })

        sorted_attributions = sorted(attributions, key=lambda x: x["abs_shap"], reverse=True)

        return {
            "base_value": base_val,
            "base_value_prob": base_prob,
            "top_reasons": sorted_attributions[:top_k],
            "all_attributions": sorted_attributions,
        }
