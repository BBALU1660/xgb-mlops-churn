
## XGB-MLOps Churn

**CI:** (add after first push)

## What is this project?

**XGB-MLOps Churn** is a production-ready, free churn prediction service built on XGBoost, packaged with MLflow (tracking + Model Registry), a FastAPI inference API, and Docker for portable deployment. It trains on the public IBM Telco Customer Churn dataset and serves probability-of-churn scores for incoming customer records.

## Why it’s useful

- **End-to-end MLOps:** experiment tracking, model versioning, stages (Staging/Production), reproducible runs.
- **Simple, fast inference:** lightweight CPU-only serving with a clean `/predict` endpoint.
- **Safe by default:** preprocessing (imputation + one-hot) and a stored model signature to prevent schema drift surprises.
- **Local-first, cloud-ready:** everything runs on your laptop; mirrors patterns used in production.

## Tech stack

- **Modeling:** XGBoost (binary classification), Optuna (hyperparameter tuning)
- **MLOps:** MLflow server + artifacts + Model Registry
- **Serving:** FastAPI + Uvicorn
- **Packaging/CI:** Docker, GitHub Actions, PyTest

## Use cases

- Churn risk scoring for retention/campaign targeting
- Batch scoring (nightly snapshots) or real-time (agent screen pops)
- Experiment management: compare AUC across parameter sweeps in MLflow
- Safe promotion: register several versions; promote the best to Production without code changes

## UI screenshots

Add your screenshots to `docs/images/` and they’ll render here:

**MLflow UI – Run & Artifacts**

<!-- ![MLflow UI](docs/images/mlflow_ui.png) -->

**Model Registry – Promotion to Production**

<!-- ![Model Registry](docs/images/model_registry.png) -->

**API – /docs (Swagger) & /health**

<!-- ![API Docs](docs/images/api_docs.png) -->

*Tip: Take these after your first successful train & serve. Filenames are placeholders—keep them or rename in the markdown.*

---

## Installation & Quickstart

Works fully offline; no paid APIs or keys.

### 0) Prerequisites


- Python 3.11
- Git
- (Optional) Docker Desktop

*Windows: enable WSL2 + virtualization (BIOS SVM/AMD-V), then install Docker Desktop.*

### 1) Clone & create a venv

```sh
git clone https://github.com/BBALU1660/xgb-mlops-churn.git
cd xgb-mlops-churn
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Get the dataset (IBM Telco Churn)

```powershell
New-Item -ItemType Directory -Force -Path data\raw | Out-Null
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" `
  -OutFile "data/raw/Telco-Customer-Churn.csv"
```

### 3) Start MLflow server with artifact serving (Terminal A)

```powershell
python -m mlflow server `
  --backend-store-uri sqlite:///mlflow.db `
  --serve-artifacts `
  --artifacts-destination "file:///$((Get-Location).Path -replace '\\','/')/mlruns" `
  --host 127.0.0.1 `
  --port 5000
```

*Keep this running.*

### 4) Train & register a model (Terminal B)

The training script creates a new experiment with `artifact_location=mlflow-artifacts:/`, logs the model with a signature, and registers a new version under `xgb_churn`.

```powershell
.\.venv\Scripts\activate
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python .\src\train.py
```

Open MLflow UI: [http://127.0.0.1:5000](http://127.0.0.1:5000)
→ check Runs & Model Registry. Promote the latest version to Production.

### 5) Serve the model (choose one)

#### A) Local (no Docker):

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
uvicorn src.serve:app --host 127.0.0.1 --port 8000 --reload
# Health: http://127.0.0.1:8000/health
# Docs:   http://127.0.0.1:8000/docs
```

#### B) Docker (talks to host MLflow):

```sh
docker build -t xgb-churn:cpu .
docker run --rm -p 18080:8080 \
  -e MODEL_URI="models:/xgb_churn/Production" \
  -e MLFLOW_TRACKING_URI="http://host.docker.internal:5000" \
  xgb-churn:cpu
# Health: http://127.0.0.1:18080/health
# Docs:   http://127.0.0.1:18080/docs
```

### 6) Call the API

```powershell
$body = @{
  rows = @(
    @{ data = @{
      gender="Female"; SeniorCitizen=0; Partner="Yes"; Dependents="No";
      tenure=1; PhoneService="No"; MultipleLines="No phone service";
      InternetService="DSL"; OnlineSecurity="No"; OnlineBackup="Yes";
      DeviceProtection="No"; TechSupport="No"; StreamingTV="No";
      StreamingMovies="No"; Contract="Month-to-month"; PaperlessBilling="Yes";
      PaymentMethod="Electronic check"; MonthlyCharges=29.85; TotalCharges=29.85
    }}
  )
} | ConvertTo-Json -Depth 4

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -Body $body -ContentType 'application/json'
```

---

## Endpoints

- `GET /health` → `{ "status": "ok", "model_uri": "models:/xgb_churn/..." }`
- `GET /model/metadata` → expected columns (from MLflow signature)
- `POST /predict` → `{ "probs": [0.12, 0.87, ...] }`

---

## Tests & CI

Run locally:

```sh
pytest -q
```

On GitHub, a basic CI (see `.github/workflows/ci.yml`) installs deps and runs tests on every push.

---

## Performance

Typical ROC-AUC on a stratified holdout is around **0.84** for the default pipeline on this dataset.
Use the scripts below to regenerate metrics & plots for your run and embed them in this README.

### Plots (saved to `docs/images/`)

- `docs/images/roc_auc.png` – ROC curve
- `docs/images/pr_curve.png` – Precision-Recall curve
- `docs/images/calibration_curve.png` – Calibration curve
- `docs/images/feature_importance.png` – XGBoost feature importance (gain)
- `docs/images/confusion_matrix.png` – Confusion matrix @ chosen threshold

We’ll add a one-click `report.py` to generate these (next step). Once generated, they’ll render here:

| ROC-AUC | PR Curve | Calibration |
|---------|----------|-------------|
|         |          |             |

| Feature Importance | Confusion Matrix |
|--------------------|-----------------|
|                    |                 |

---

## Notes & Troubleshooting

- If the container fails with a Windows file path inside the error, you trained before enabling `--serve-artifacts`. Create a new experiment with `artifact_location=mlflow-artifacts:/` (the training script already does this), then retrain and promote the new version.
- If Docker can’t bind port 8080, map a different host port (e.g., `-p 18080:8080`).
- On Windows, for Docker Desktop you need WSL2 + virtualization enabled (BIOS SVM/AMD-V).
