import pandas as pd
import joblib

model=joblib.load("model/churn_model.pkl")
feature=joblib.load("model/feature_columns.pkl")

def predict_churn(data:dict):
    df=pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature, fill_value=0)
    prediction=model.predict(df)
    return "Yes" if prediction[0]==1 else "No"