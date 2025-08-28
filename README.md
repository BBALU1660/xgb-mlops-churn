# XGBoost MLOps — Telco Churn

Train an XGBoost classifier with MLflow tracking + Optuna HPO.
Serve the registered model via FastAPI. CPU-only, no API keys.

## Structure
- src/train.py — train, tune, log to MLflow, register model
- src/serve.py — FastAPI service loading MLflow model
- src/schema.py — request/response models
- tests/test_smoke.py — quick sanity test

## Quickstart

### Train locally (logs to MLflow + registers a version)
```powershell
# Terminal A — MLflow server with artifact serving
python -m mlflow server `
  --backend-store-uri sqlite:///mlflow.db `
  --serve-artifacts `
  --artifacts-destination "file:///<ABSOLUTE_PATH_TO>/xgb-mlops-churn/mlruns" `
  --host 127.0.0.1 --port 5000

# Terminal B — train (new experiment uses mlflow-artifacts:/)
.\.venv\Scripts\activate
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python .\src\train.py
