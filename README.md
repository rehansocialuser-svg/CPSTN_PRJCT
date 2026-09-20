# E-Commerce Return Risk & Cost-Sensitive Intervention Engine

An end-to-end operational ML framework that predicts transaction-level return risk and evaluates an asymmetric cost matrix to minimize total reverse-logistics loss.

---

## 1. System Architecture (Partner Split)

- **Partner A (Complete):** Tabular preprocessing pipeline, LightGBM classifier with class imbalance re-weighting, and decision-theoretic cost-sensitive threshold optimizer.
- **Partner B (Pending):** NLP Reason Extraction (Aspect classification from customer feedback) and FastAPI + Streamlit integration dashboard.

---

## 2. Business Cost Matrix & Threshold Tuning

Standard classification at $t = 0.50$ failed catastrophically on imbalanced return distribution (Recall = 0.00%). The decision boundary was re-calibrated by minimizing the expected cost function:

$$\text{Loss} = (FP \cdot C_{FP}) + (FN \cdot C_{FN}) + (TP \cdot C_{TP}) + (TN \cdot C_{TN})$$

- **$C_{FN}$ (False Negative):** INR 260.00 (Unmitigated return logistics + restocking)
- **$C_{FP}$ (False Positive):** INR 40.00 (Unnecessary outreach/friction voucher)
- **$C_{TP}$ (True Positive):** INR 55.00 (Mitigation outreach expense)
- **$C_{TN}$ (True Negative):** INR 0.00

### Empirical Evaluation on Holdout Test Set (3,000 Orders)
| Metric | Baseline Policy ($t = 0.50$) | Optimized Policy ($t^* = 0.2900$) |
| :--- | :--- | :--- |
| **Return Recall** | 0.00% | **90.42%** |
| **Precision** | 0.00% | 30.09% |
| **Total Loss** | INR 211,640.00 | **INR 129,160.00** |
| **Net Financial Savings** | INR 0.00 | **INR 82,480.00** |

---

## 3. Partner B Handoff Contract

### Setup
```bash
git clone [https://github.com/rehansocialuser-svg/CPSTN_PRJCT.git](https://github.com/rehansocialuser-svg/CPSTN_PRJCT.git)
cd CPSTN_PRJCT
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
from src.inference_engine import ReturnRiskInferenceEngine

engine = ReturnRiskInferenceEngine(models_dir="models")

order_payload = {
    "order_id": "ORD_1024",
    "category": "Apparel",
    "order_amount": 2499.0,
    "delivery_delay_days": 3,
    "is_cod": 1,
    "customer_past_return_rate": 0.35,
    "product_weight_g": 600.0,
    "installments": 1
}

decision = engine.predict_order(order_payload)
print(decision)
# Output:
# {
#   'order_id': 'ORD_1024',
#   'return_probability': 0.4008,
#   'optimal_threshold': 0.29,
#   'action': 'INTERVENE',
#   'risk_tier': 'MEDIUM',
#   'expected_savings_inr': 58.2
# }