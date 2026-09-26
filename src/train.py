import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report
)
from feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "risk_model.joblib")
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "baseline_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")

def train_and_compare_models():
    # 1. Load Data & Preprocessor
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    
    X_train_raw = train_df[features]
    y_train = train_df[TARGET_COL].values
    
    X_test_raw = test_df[features]
    y_test = test_df[TARGET_COL].values
    
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_weight = float(num_neg) / float(num_pos)
    
    # -------------------------------------------------------------
    # 2. BASELINE MODEL: Weighted Logistic Regression
    # -------------------------------------------------------------
    baseline_model = LogisticRegression(class_weight="balanced", random_state=42)
    baseline_model.fit(X_train, y_train)
    base_probs = baseline_model.predict_proba(X_test)[:, 1]
    
    base_roc = roc_auc_score(y_test, base_probs)
    base_pr = average_precision_score(y_test, base_probs)
    
    # -------------------------------------------------------------
    # 3. ADVANCED MODEL: LightGBM Classifier
    # -------------------------------------------------------------
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
    
    lgb_roc = roc_auc_score(y_test, lgb_probs)
    lgb_pr = average_precision_score(y_test, lgb_probs)
    lgb_brier = brier_score_loss(y_test, lgb_probs)
    
    # -------------------------------------------------------------
    # 4. MANDATORY BASELINE VS ADVANCED COMPARISON REPORT
    # -------------------------------------------------------------
    print("\n=============================================================")
    print("      MANDATORY MODEL BENCHMARK & BASELINE COMPARISON        ")
    print("=============================================================")
    print(f"{'Metric':<25} | {'Baseline (Logistic)':<20} | {'Advanced (LightGBM)':<20}")
    print("-" * 70)
    print(f"{'ROC-AUC Score':<25} | {base_roc:<20.4f} | {lgb_roc:<20.4f}")
    print(f"{'PR-AUC Score (Avg Prec)':<25} | {base_pr:<20.4f} | {lgb_pr:<20.4f}")
    print(f"{'Brier Calibration Loss':<25} | {'N/A':<20} | {lgb_brier:<20.4f}")
    print("=============================================================\n")
    
    # 5. Extract Feature Importance (Explainability requirement)
    feature_names = preprocessor.get_feature_names_out()
    importances = lgb_model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_gain": importances
    }).sort_values(by="importance_gain", ascending=False)
    
    print("--- MODEL EXPLAINABILITY: TOP FEATURE IMPORTANCES ---")
    print(importance_df.head(6).to_string(index=False))
    print("-----------------------------------------------------\n")
    
    # Save Artifacts
    joblib.dump(baseline_model, BASELINE_MODEL_PATH)
    joblib.dump(lgb_model, MODEL_PATH)
    
    test_results_df = test_df.copy()
    test_results_df["predicted_return_prob"] = lgb_probs
    test_results_df.to_csv("data/processed/test_predictions.csv", index=False)
    print("[SUCCESS] All baseline and advanced artifacts serialized to models/")

if __name__ == "__main__":
    train_and_compare_models()