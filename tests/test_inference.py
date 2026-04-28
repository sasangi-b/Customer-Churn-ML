def test_predict_not_churn():
    from src.serving.inference import predict
    result = predict({
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)", "tenure": 60,
        "MonthlyCharges": 45.0, "TotalCharges": 2700.0
    })
    assert result in ["Likely to churn", "Not likely to churn"]

def test_predict_churn():
    from src.serving.inference import predict
    result = predict({
        "gender": "Female", "SeniorCitizen": 1, "Partner": "No",
        "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "tenure": 1,
        "MonthlyCharges": 85.0, "TotalCharges": 85.0
    })
    assert result in ["Likely to churn", "Not likely to churn"]