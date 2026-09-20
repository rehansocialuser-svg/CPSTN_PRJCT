import sys
import os

# Add src to system path
sys.path.append(os.path.abspath("src"))

from inference_engine import ReturnRiskInferenceEngine

def main():
    print("Testing ReturnRiskInferenceEngine handshake...\n")
    engine = ReturnRiskInferenceEngine(models_dir="models")
    
    # Test Case 1: High Risk Order (COD, Apparel, Delayed, Customer with high return history)
    order_high_risk = {
        "order_id": "ORD_TEST_HIGH_01",
        "category": "Apparel",
        "order_amount": 2499.0,
        "delivery_delay_days": 4,
        "is_cod": 1,
        "customer_past_return_rate": 0.45,
        "product_weight_g": 650.0,
        "installments": 1
    }
    
    # Test Case 2: Low Risk Order (Prepaid, Beauty, Fast delivery, Loyal customer)
    order_low_risk = {
        "order_id": "ORD_TEST_LOW_02",
        "category": "Beauty",
        "order_amount": 799.0,
        "delivery_delay_days": 0,
        "is_cod": 0,
        "customer_past_return_rate": 0.02,
        "product_weight_g": 220.0,
        "installments": 1
    }
    
    res1 = engine.predict_order(order_high_risk)
    res2 = engine.predict_order(order_low_risk)
    
    print("--- Test Case 1 (High Risk Scenario) ---")
    print(res1)
    
    print("\n--- Test Case 2 (Low Risk Scenario) ---")
    print(res2)
    
    assert res1["action"] == "INTERVENE", "High risk order should trigger INTERVENE"
    assert res2["action"] == "NO_ACTION", "Low risk order should trigger NO_ACTION"
    print("\n[VERIFICATION PASSED] Partner A core pipeline is 100% complete and ready for Partner B.")

if __name__ == "__main__":
    main()