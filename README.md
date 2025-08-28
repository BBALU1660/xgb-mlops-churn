# XGBoost MLOps — Telco Churn

Train an XGBoost classifier with MLflow tracking + Optuna HPO.
Serve the registered model via FastAPI. CPU-only, no API keys.

## Structure
- src/train.py — train, tune, log to MLflow, register model
- src/serve.py — FastAPI service loading MLflow model
- src/schema.py — request/response models
- tests/test_smoke.py — quick sanity test
