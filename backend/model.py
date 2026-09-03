"""
model.py — Model Artifact Loader for RiskLedger.
Loads trained XGBoost model & scaler from joblib artifact without retraining on request.
"""

import joblib
from pathlib import Path
from backend.explain import RiskExplainer

MODEL_PATH = Path(__file__).parent / "xgb_model.joblib"


class ModelService:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.xgb_model = None
        self.scaler = None
        self.logreg = None
        self.features = None
        self.threshold = 0.13
        self.category_policies = {}
        self.explainer = None
        self.load_model()

    def load_model(self):
        if not self.model_path.exists():
            print(f"[ModelService] {self.model_path} missing. Running training pipeline...")
            from backend.train import main as run_training
            run_training()

        saved_data = joblib.load(self.model_path)
        self.xgb_model = saved_data["model"]
        self.scaler = saved_data["scaler"]
        self.logreg = saved_data["logreg"]
        self.features = saved_data["features"]
        self.threshold = float(saved_data.get("threshold", 0.13))
        self.category_policies = saved_data.get("category_policies", {})

        self.explainer = RiskExplainer(self.xgb_model, self.features)
        print(f"[ModelService] Loaded production XGBoost model. Validation-locked threshold: {self.threshold:.3f}")


_model_service = None

def get_model_service():
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
