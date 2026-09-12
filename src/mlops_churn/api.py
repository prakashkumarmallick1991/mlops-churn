import mlflow
import mlflow.sklearn

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
)


MODEL_URI = "models:/churn_model@champion"

model = mlflow.sklearn.load_model(MODEL_URI)


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

    features = [[
        customer.monthly_charges,
        customer.tenure,
        customer.support_calls,
        customer.login_frequency,
        customer.contract_length,
    ]]

    prediction = model.predict(features)[0]

    return {
        "churn": int(prediction)
    }