import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

MODELS_DIR = "models"
NUMERIC_FEATURES = [
    "order_amount",
    "delivery_delay_days",
    "customer_past_return_rate",
    "product_weight_g",
    "installments"
]
CATEGORICAL_FEATURES = [
    "category",
    "is_cod"
]
TARGET_COL = "is_returned"

def get_preprocessor() -> ColumnTransformer:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERIC_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

def fit_and_save_preprocessor(train_df_path: str = "data/processed/train.csv") -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(train_df_path)
    
    X_train = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preprocessor = get_preprocessor()
    preprocessor.fit(X_train)
    
    save_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, save_path)
    print(f"[SUCCESS] Preprocessor fitted and saved to {save_path}")
    
    # Feature count validation
    transformed_sample = preprocessor.transform(X_train.head(5))
    print(f"Transformed feature dimension: {transformed_sample.shape[1]} columns")
    return save_path

if __name__ == "__main__":
    fit_and_save_preprocessor()