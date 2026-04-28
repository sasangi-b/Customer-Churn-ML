"""
tests/test_inference.py
-----------------------
Mocks mlflow before inference.py loads so CI works
without mlruns/ or a real model file.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock

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

NOT_CHURN_CUSTOMER = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "Yes", "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)", "tenure": 60,
    "MonthlyCharges": 45.0, "TotalCharges": 2700.0
}

CHURN_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 1, "Partner": "No",
    "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "tenure": 1,
    "MonthlyCharges": 85.0, "TotalCharges": 85.0
}


def get_inference_module(predict_return: int):
    """
    Reload inference module fresh with mlflow fully mocked.
    Mocking must happen before the module loads since model
    is loaded at import time (module-level code).
    """
    # Clear any cached version of the module
    for key in list(sys.modules.keys()):
        if "src.serving.inference" in key:
            del sys.modules[key]

    # Create a fake model
    mock_model = MagicMock()
    mock_model.predict.return_value = [predict_return]

    # Patch mlflow and glob before the module loads
    with patch("mlflow.pyfunc.load_model", return_value=mock_model), \
         patch("glob.glob", return_value=["./mlruns/fake/models/fake/artifacts"]), \
         patch("os.path.exists", return_value=False), \
         patch("os.path.getmtime", return_value=0):
        import src.serving.inference as inf
        # Override module-level globals with our mocks
        inf.model = mock_model
        inf.FEATURE_COLS = FEATURE_COLS
        return inf


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_predict_not_churn():
    inf = get_inference_module(predict_return=0)
    assert inf.predict(NOT_CHURN_CUSTOMER) == "Not likely to churn"


def test_predict_churn():
    inf = get_inference_module(predict_return=1)
    assert inf.predict(CHURN_CUSTOMER) == "Likely to churn"


def test_predict_returns_string():
    inf = get_inference_module(predict_return=0)
    assert isinstance(inf.predict(NOT_CHURN_CUSTOMER), str)


def test_predict_valid_labels():
    inf = get_inference_module(predict_return=0)
    result = inf.predict(NOT_CHURN_CUSTOMER)
    assert result in ["Likely to churn", "Not likely to churn"]


def test_senior_citizen_as_int():
    inf = get_inference_module(predict_return=0)
    data = {**NOT_CHURN_CUSTOMER, "SeniorCitizen": 1}
    assert inf.predict(data) in ["Likely to churn", "Not likely to churn"]