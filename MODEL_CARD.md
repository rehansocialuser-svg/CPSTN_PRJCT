# Model Card: E-Commerce Return Risk Predictor (LGBM-R1)

## 1. Model Details
- **Model Architecture:** LightGBM Classifier (Gradient Boosted Decision Trees) + Logistic Regression Baseline
- **Version:** 1.0.0
- **Input Features:** 8 features (`category`, `order_amount`, `delivery_delay_days`, `is_cod`, `customer_past_return_rate`, `product_weight_g`, `installments`)
- **Target Variable:** `is_returned` (Binary: 0 = Retained, 1 = Returned)

## 2. Intended Use
- **Primary Use Case:** Real-time transaction-level return probability scoring to trigger cost-optimal post-order interventions.
- **Out of Scope:** Automated order cancellation or credit scoring without human oversight.

## 3. Training & Evaluation Data
- **Dataset Size:** 15,000 transaction records (12,000 Train / 3,000 Holdout Test).
- **Target Class Balance:** ~27.1% Return Rate (Imbalanced). Re-balanced via `scale_pos_weight`.

## 4. Operational Performance Benchmark
- **Logistic Regression Baseline ROC-AUC:** 0.6670
- **LightGBM ROC-AUC:** 0.6431
- **Cost-Optimal Threshold ($t^*$):** 0.2900 (Minimizes expected reverse logistics loss)