import os
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from src.schema import PredictRequest, PredictResponse

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(TRACKING_URI)

MODEL_URI = os.getenv("MODEL_URI", "models:/xgb_churn/Production")
model = mlflow.sklearn.load_model(MODEL_URI)

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = pd.DataFrame([r.data for r in req.rows])
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(df)[:, 1]
    else:
        p = model.predict(df)
    return PredictResponse(probs=[float(x) for x in p])
