import mlflow
import mlflow.sklearn
import pandas as pd
import os

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
)


MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/churn_model@champion",
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_model():
    return mlflow.sklearn.load_model(MODEL_URI)


model = load_model()


class CustomerInput(BaseModel):
    monthly_charges: float
    tenure: float
    support_calls: float
    login_frequency: float
    contract_length: float


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(customer: CustomerInput):

    features = pd.DataFrame([{
    "monthly_charges": customer.monthly_charges,
    "tenure": customer.tenure,
    "support_calls": customer.support_calls,
    "login_frequency": customer.login_frequency,
    "contract_length": customer.contract_length,
}])

    prediction = model.predict(features)[0]

    return {
        "churn": int(prediction)
    }