"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================
Fixed for local development: robust model discovery from mlruns/
"""

import os
import glob
import pandas as pd
import mlflow

# === MODEL LOADING ===
# Tries Docker path first, then discovers automatically from mlruns/

def _find_local_model():
    """Search mlruns recursively for the most recently modified model folder."""
    # Matches your actual structure:
    # mlruns/<experiment_id>/models/<model_id>/artifacts/
    patterns = [
        "./mlruns/*/models/*/artifacts",        # your actual structure
        "./mlruns/*/*/artifacts/model",          # standard mlflow structure
        "./mlruns/*/*/*/artifacts/model",        # nested standard structure
        "../mlruns/*/models/*/artifacts",        # parent directory variants
        "../mlruns/*/*/artifacts/model",
    ]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))

    if not found:
        raise FileNotFoundError(
            "No model found in mlruns/. "
            "Make sure you have run the training pipeline at least once."
        )

    # Return the most recently modified (i.e. latest run)
    return max(found, key=os.path.getmtime)


def _find_feature_columns(model_dir):
    """
    Look for feature_columns.txt in multiple likely locations:
    - Inside the model folder itself
    - In the artifacts folder (sibling of model/)
    - In the project root
    - In src/serving/
    """
    candidates = [
        os.path.join(model_dir, "feature_columns.txt"),
        os.path.join(os.path.dirname(model_dir), "feature_columns.txt"),  # artifacts/
        "./feature_columns.txt",
        "./src/serving/feature_columns.txt",
        "./artifacts/feature_columns.txt",
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"✅ Found feature_columns.txt at: {path}")
            return path
    return None


# --- Load model ---
MODEL_DIR = "/app/model"  # Docker production path

if os.path.exists(MODEL_DIR):
    try:
        model = mlflow.pyfunc.load_model(MODEL_DIR)
        print(f"✅ Model loaded from Docker path: {MODEL_DIR}")
    except Exception as e:
        print(f"❌ Docker path exists but failed to load: {e}")
        MODEL_DIR = None
else:
    MODEL_DIR = None

if MODEL_DIR is None:
    print("🔍 Docker model not found, searching mlruns/ ...")
    try:
        MODEL_DIR = _find_local_model()
        model = mlflow.pyfunc.load_model(MODEL_DIR)
        print(f"✅ Model loaded from: {MODEL_DIR}")
    except Exception as e:
        raise RuntimeError(
            f"Could not load model. Details: {e}\n\n"
            "Fix: Run your training pipeline first, or check that mlruns/ "
            "exists in your project root directory."
        )

# --- Load feature columns ---
feature_file = _find_feature_columns(MODEL_DIR)

if feature_file:
    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns")
else:
    # ── FALLBACK: hardcoded feature columns ──────────────────────────────────
    # These are derived from the Telco dataset with drop_first=True OHE.
    # Used only if feature_columns.txt was never saved during training.
    # To fix permanently: save feature columns in your training script (see below).
    print("⚠️  feature_columns.txt not found — using hardcoded fallback columns.")
    print("   To fix: add this to your training script after fitting:")
    print("       with open('feature_columns.txt', 'w') as f:")
    print("           f.write('\\n'.join(X_train.columns.tolist()))")
    FEATURE_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "OnlineBackup_No internet service", "OnlineBackup_Yes",
    "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_One year", "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]

# === FEATURE TRANSFORMATION CONSTANTS ===
BINARY_MAP = {
    "gender":           {"Female": 0, "Male": 1},
    "Partner":          {"No": 0, "Yes": 1},
    "Dependents":       {"No": 0, "Yes": 1},
    "PhoneService":     {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Step 1: Numeric coercion
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Step 2: Binary encoding
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c].astype(str).str.strip()
                .map(mapping).astype("Int64").fillna(0).astype(int)
            )

    # Step 3: One-hot encode remaining categoricals
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # Step 4: Bool → int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)

    # Step 5: Align to training feature schema
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    return df


def predict(input_dict: dict) -> str:
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    try:
        preds = model.predict(df_enc)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        result = preds[0] if isinstance(preds, (list, tuple)) and len(preds) == 1 else preds
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    return "Likely to churn" if result == 1 else "Not likely to churn"