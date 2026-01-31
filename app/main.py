from fastapi import FastAPI
from app.schema import CustomerData
from app.model_loader import predict_churn

app=FastAPI(title="Customer Churn Prediction API")

@app.get("/")
def home():
    return {"message":"Welcome to the Customer Churn Prediction API"}

@app.post("/predict")
def predict(data:CustomerData):
    result=predict_churn(data.dict())
    return {"churn_prediction":result}