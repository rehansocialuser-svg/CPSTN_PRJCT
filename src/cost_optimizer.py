import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Business Cost Matrix (in INR)
# Cost of False Positive: Unnecessary friction / wasted incentive voucher
COST_FP = 40.0
# Cost of False Negative: Unmitigated return (reverse logistics + restocking + depreciation)
COST_FN = 260.0
# Cost of True Positive: Successful intervention cost (e.g. outreach + sizing confirmation)
COST_TP = 55.0
# Cost of True Negative: Normal delivery, clean transaction
COST_TN = 0.0

class CostOptimizer:
    def __init__(self, c_fp=COST_FP, c_fn=COST_FN, c_tp=COST_TP, c_tn=COST_TN):
        self.c_fp = c_fp
        self.c_fn = c_fn
        self.c_tp = c_tp
        self.c_tn = c_tn
        self.optimal_threshold = 0.5
        self.min_cost = None

    def compute_cost(self, y_true: np.ndarray, y_probs: np.ndarray, threshold: float) -> dict:
        y_pred = (y_probs >= threshold).astype(int)
        
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        
        total_cost = (fp * self.c_fp) + (fn * self.c_fn) + (tp * self.c_tp) + (tn * self.c_tn)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {
            "threshold": threshold,
            "total_cost": total_cost,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision,
            "recall": recall
        }

    def find_optimal_threshold(self, y_true: np.ndarray, y_probs: np.ndarray):
        thresholds = np.linspace(0.05, 0.95, 181)
        results = [self.compute_cost(y_true, y_probs, t) for t in thresholds]
        
        best = min(results, key=lambda x: x["total_cost"])
        self.optimal_threshold = best["threshold"]
        self.min_cost = best["total_cost"]
        
        # Calculate baseline cost at standard threshold 0.5
        baseline_05 = self.compute_cost(y_true, y_probs, 0.50)
        # Calculate worst-case cost (no intervention at all)
        no_action_cost = int(np.sum(y_true == 1)) * self.c_fn
        
        return best, baseline_05, no_action_cost, results

def run_optimization():
    test_pred_path = "data/processed/test_predictions.csv"
    if not os.path.exists(test_pred_path):
        raise FileNotFoundError(f"{test_pred_path} not found. Run train.py first.")
        
    df = pd.read_csv(test_pred_path)
    y_true = df["is_returned"].values
    y_probs = df["predicted_return_prob"].values
    
    optimizer = CostOptimizer()
    best, baseline, no_action, curve = optimizer.find_optimal_threshold(y_true, y_probs)
    
    savings_vs_baseline = baseline["total_cost"] - best["total_cost"]
    savings_vs_noaction = no_action - best["total_cost"]
    
    print("\n================ BUSINESS COST OPTIMIZATION RESULTS ================")
    print(f"No-Intervention Policy Total Loss      : INR {no_action:,.2f}")
    print(f"Default 0.50 Threshold Total Cost      : INR {baseline['total_cost']:,.2f}")
    print(f"  -> Recall at 0.50                    : {baseline['recall']:.2%}")
    print(f"  -> Precision at 0.50                 : {baseline['precision']:.2%}")
    print("---------------------------------------------------------------------")
    print(f"OPTIMAL BUSINESS THRESHOLD (t*)        : {best['threshold']:.4f}")
    print(f"Optimized Policy Total Cost            : INR {best['total_cost']:,.2f}")
    print(f"  -> Captured Returns (Recall at t*)   : {best['recall']:.2%}")
    print(f"  -> Precision at t*                   : {best['precision']:.2%}")
    print(f"  -> TP: {best['tp']} | FP: {best['fp']} | FN: {best['fn']} | TN: {best['tn']}")
    print("---------------------------------------------------------------------")
    print(f"NET PROFIT SAVED VS DEFAULT 0.50       : INR {savings_vs_baseline:,.2f}")
    print(f"NET PROFIT SAVED VS NO INTERVENTION    : INR {savings_vs_noaction:,.2f}")
    print("=====================================================================\n")
    
    # Save optimizer config for Partner B's production inference
    opt_payload = {
        "optimal_threshold": float(best["threshold"]),
        "cost_matrix": {
            "c_fp": COST_FP,
            "c_fn": COST_FN,
            "c_tp": COST_TP,
            "c_tn": COST_TN
        },
        "expected_recall": float(best["recall"]),
        "expected_precision": float(best["precision"])
    }
    
    save_path = "models/cost_policy.joblib"
    joblib.dump(opt_payload, save_path)
    print(f"[SUCCESS] Cost policy serialized at: {save_path}")

if __name__ == "__main__":
    run_optimization()