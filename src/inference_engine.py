import os
import joblib
import pandas as pd
from typing import Dict, Any

class ReturnRiskInferenceEngine:
    def __init__(self, models_dir: str = "models"):
        self.preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
        self.model_path = os.path.join(models_dir, "risk_model.joblib")
        self.policy_path = os.path.join(models_dir, "cost_policy.joblib")
        
        self._load_artifacts()

    def _load_artifacts(self):
        if not all(os.path.exists(p) for p in [self.preprocessor_path, self.model_path, self.policy_path]):
            raise FileNotFoundError("Required model artifacts missing in models/ directory.")
            
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)
        self.policy = joblib.load(self.policy_path)
        self.optimal_threshold = self.policy["optimal_threshold"]

    def predict_order(self, order_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts single order features and returns business action.
        """
        df_single = pd.DataFrame([order_dict])
        
        # Transform using preprocessor
        X_trans = self.preprocessor.transform(df_single)
        
        # Compute return probability
        prob = float(self.model.predict_proba(X_trans)[:, 1][0])
        
        # Apply cost-sensitive threshold decision
        should_intervene = bool(prob >= self.optimal_threshold)
        
        # Expected business impact calculation
        cost_matrix = self.policy["cost_matrix"]
        expected_cost_without_action = prob * cost_matrix["c_fn"]
        expected_cost_with_action = (
            prob * cost_matrix["c_tp"] + (1.0 - prob) * cost_matrix["c_fp"]
        )
        
        expected_savings = max(0.0, expected_cost_without_action - expected_cost_with_action)
        
        return {
            "order_id": order_dict.get("order_id", "UNKNOWN"),
            "return_probability": round(prob, 4),
            "optimal_threshold": self.optimal_threshold,
            "action": "INTERVENE" if should_intervene else "NO_ACTION",
            "risk_tier": "HIGH" if prob >= 0.50 else ("MEDIUM" if prob >= self.optimal_threshold else "LOW"),
            "expected_savings_inr": round(expected_savings, 2) if should_intervene else 0.0
        }