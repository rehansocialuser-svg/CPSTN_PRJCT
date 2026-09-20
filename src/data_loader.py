import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = os.path.join("data", "raw", "ecommerce_returns.csv")
PROCESSED_DATA_DIR = os.path.join("data", "processed")

def generate_synthetic_dataset(n_samples: int = 15000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    order_ids = [f"ORD_{100000 + i}" for i in range(n_samples)]
    customer_ids = [f"CUST_{np.random.randint(1000, 5000)}" for _ in range(n_samples)]
    
    # Categories with baseline return risks
    categories = ['Apparel', 'Footwear', 'Electronics', 'Home & Kitchen', 'Beauty']
    category_weights = [0.35, 0.20, 0.20, 0.15, 0.10]
    category_risk_map = {
        'Apparel': 0.30,
        'Footwear': 0.25,
        'Electronics': 0.12,
        'Home & Kitchen': 0.10,
        'Beauty': 0.08
    }
    
    assigned_categories = np.random.choice(categories, size=n_samples, p=category_weights)
    cat_risk = np.array([category_risk_map[c] for c in assigned_categories])
    
    order_amount = np.random.exponential(scale=1200, size=n_samples) + 299.0
    delivery_delay_days = np.random.choice([0, 1, 2, 3, 4, 5, 7, 10], size=n_samples, p=[0.50, 0.20, 0.12, 0.08, 0.04, 0.03, 0.02, 0.01])
    is_cod = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55])
    customer_past_return_rate = np.clip(np.random.beta(a=2, b=8, size=n_samples), 0.0, 1.0)
    product_weight_g = np.random.uniform(150.0, 3500.0, size=n_samples)
    installments = np.random.choice([1, 2, 3, 6, 12], size=n_samples, p=[0.7, 0.1, 0.1, 0.07, 0.03])
    
    # Latent probability calculation (Logistic Log-Odds)
    log_odds = (
        - 2.8
        + 2.2 * cat_risk
        + 0.15 * delivery_delay_days
        + 0.65 * is_cod
        + 3.5 * customer_past_return_rate
        + 0.00015 * order_amount
        - 0.00008 * product_weight_g
    )
    
    # Sigmoid function
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    # Binary return outcome
    is_returned = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": customer_ids,
        "category": assigned_categories,
        "order_amount": np.round(order_amount, 2),
        "delivery_delay_days": delivery_delay_days,
        "is_cod": is_cod,
        "customer_past_return_rate": np.round(customer_past_return_rate, 4),
        "product_weight_g": np.round(product_weight_g, 1),
        "installments": installments,
        "is_returned": is_returned
    })
    
    return df

def save_and_split_data():
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    df = generate_synthetic_dataset()
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"[SUCCESS] Raw dataset generated at {RAW_DATA_PATH} with shape: {df.shape}")
    print(f"Class distribution: Return rate = {df['is_returned'].mean():.2%}")
    
    # Stratified Train/Test split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['is_returned'], random_state=42)
    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[SUCCESS] Train saved to {train_path} ({train_df.shape})")
    print(f"[SUCCESS] Test saved to {test_path} ({test_df.shape})")

if __name__ == "__main__":
    save_and_split_data()