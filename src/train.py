import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    brier_score_loss
)
from feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "risk_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")

def train_model():
    # 1. Load data & preprocessor
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"{PREPROCESSOR_PATH} not found. Run feature_engineering.py first.")
    
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train_raw = train_df[features]
    y_train = train_df[TARGET_COL].values
    
    X_test_raw = test_df[features]
    y_test = test_df[TARGET_COL].values
    
    # 2. Transform tabular features
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # 3. Calculate class imbalance weight
    num_negatives = np.sum(y_train == 0)
    num_positives = np.sum(y_train == 1)
    scale_weight = float(num_negatives) / float(num_positives)
    print(f"Class Distribution: {num_negatives} Negatives vs {num_positives} Positives")
    print(f"Calculated scale_pos_weight: {scale_weight:.3f}")
    
    # 4. Train LightGBM Classifier
    model = lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        random_state=42,
        importance_type="gain"
    )
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    # 5. Evaluate on Holdout Test Set
    test_probs = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, test_probs)
    pr_auc = average_precision_score(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)
    
    print("\n--- MODEL EVALUATION METRICS (TEST SET) ---")
    print(f"ROC-AUC Score          : {roc_auc:.4f}")
    print(f"PR-AUC Score (Avg Prec): {pr_auc:.4f}")
    print(f"Brier Score (Calibr.)  : {brier:.4f}")
    
    # Baseline 0.5 threshold report
    print("\n--- BASELINE (0.5 THRESHOLD) CLASSIFICATION REPORT ---")
    print(classification_report(y_test, (test_probs >= 0.5).astype(int), digits=4))
    
    # 6. Save Model Artifact
    joblib.dump(model, MODEL_PATH)
    print(f"[SUCCESS] Trained risk model serialized at: {MODEL_PATH}")
    
    # 7. Save test predictions for Step 6 (Cost Optimization)
    test_results_df = test_df.copy()
    test_results_df["predicted_return_prob"] = test_probs
    test_results_df.to_csv("data/processed/test_predictions.csv", index=False)
    print("[SUCCESS] Exported data/processed/test_predictions.csv for cost optimizer.")

if __name__ == "__main__":
    train_model()