import os
import pytest
import pandas as pd
import numpy as np
import joblib
from src.inference_engine import ReturnRiskInferenceEngine

MODELS_DIR = "models"

@pytest.fixture
def inference_engine():
    assert os.path.exists(os.path.join(MODELS_DIR, "risk_model.joblib")), "Model artifact missing"
    assert os.path.exists(os.path.join(MODELS_DIR, "preprocessor.joblib")), "Preprocessor missing"
    return ReturnRiskInferenceEngine(models_dir=MODELS_DIR)

def test_inference_schema_and_output(inference_engine):
    valid_payload = {
        "order_id": "TEST_001",
        "category": "Apparel",
        "order_amount": 1500.0,
        "delivery_delay_days": 2,
        "is_cod": 1,
        "customer_past_return_rate": 0.40,
        "product_weight_g": 500.0,
        "installments": 1
    }
    
    result = inference_engine.predict_order(valid_payload)
    
    # Validation checks
    assert result["order_id"] == "TEST_001"
    assert 0.0 <= result["return_probability"] <= 1.0
    assert result["action"] in ["INTERVENE", "NO_ACTION"]
    assert result["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]

def test_missing_field_error_handling(inference_engine):
    invalid_payload = {
        "order_id": "TEST_INVALID",
        "category": "Apparel"
        # Missing required operational features
    }
    
    with pytest.raises(Exception):
        inference_engine.predict_order(invalid_payload)