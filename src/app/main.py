"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok"}

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int         # 0 or 1
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def get_prediction(data: CustomerData):
    try:
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}


# === GRADIO WEB INTERFACE ===
# IMPORTANT: parameter order here must exactly match the inputs=[] list below
def gradio_interface(
    gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    data = {
        "gender": gender,
        "SeniorCitizen": int(SeniorCitizen),
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }
    result = predict(data)
    return str(result)

# === GRADIO UI ===
# INPUT ORDER must match gradio_interface() parameter order exactly
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        # 1. gender
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        # 2. SeniorCitizen  ← moved to position 2 to match function signature
        gr.Dropdown([0, 1], label="Senior Citizen (0=No, 1=Yes)", value=0),
        # 3. Partner
        gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
        # 4. Dependents
        gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),
        # 5. PhoneService
        gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
        # 6. MultipleLines
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No"),
        # 7. InternetService
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic"),
        # 8. OnlineSecurity
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No"),
        # 9. OnlineBackup
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No"),
        # 10. DeviceProtection
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No"),
        # 11. TechSupport
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No"),
        # 12. StreamingTV
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes"),
        # 13. StreamingMovies
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes"),
        # 14. Contract
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"),
        # 15. PaperlessBilling
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes"),
        # 16. PaymentMethod
        gr.Dropdown([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], label="Payment Method", value="Electronic check"),
        # 17. tenure
        gr.Number(label="Tenure (months)", value=1, minimum=0, maximum=100),
        # 18. MonthlyCharges
        gr.Number(label="Monthly Charges ($)", value=85.0, minimum=0, maximum=200),
        # 19. TotalCharges
        gr.Number(label="Total Charges ($)", value=85.0, minimum=0, maximum=10000),
    ],
    outputs=gr.Textbox(label="Churn Prediction", lines=2),
    title="🔮 Telco Customer Churn Predictor",
    description="""
    **Predict customer churn probability using machine learning**
    
    Fill in the customer details below to get a churn prediction. The model uses XGBoost trained on 
    historical telecom customer data to identify customers at risk of churning.
    
    💡 **Tip**: Month-to-month contracts with fiber optic internet and electronic check payments 
    tend to have higher churn rates.
    """,
    examples=[
        # High churn risk: gender, SeniorCitizen, Partner, Dependents, PhoneService,
        #                  MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
        #                  DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
        #                  Contract, PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
        ["Female", 1, "No", "No", "Yes", "No", "Fiber optic", "No", "No",
         "No", "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check",
         1, 85.0, 85.0],
        # Low churn risk
        ["Male", 0, "Yes", "Yes", "Yes", "Yes", "DSL", "Yes", "Yes",
         "Yes", "Yes", "No", "No", "Two year", "No", "Credit card (automatic)",
         60, 45.0, 2700.0]
    ],
)

app = gr.mount_gradio_app(app, demo, path="/ui")