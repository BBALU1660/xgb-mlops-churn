import os, pandas as pd, numpy as np, joblib, optuna
import mlflow, mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# ---------- MLflow tracking & experiment (MODULE LEVEL) ----------
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(TRACKING_URI)

EXPERIMENT_NAME = "xgb_churn_experiment_http"  # NEW experiment so artifact_uri is mlflow-artifacts:/
client = MlflowClient(tracking_uri=TRACKING_URI)
if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
    client.create_experiment(EXPERIMENT_NAME, artifact_location="mlflow-artifacts:/")
mlflow.set_experiment(EXPERIMENT_NAME)

# ---------- Data ----------
def load_data(p):
    df = pd.read_csv(p)
    df = df.replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    y = df["Churn"].values
    X = df.drop(columns=["Churn", "customerID"])
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return X, y, num, cat

# ---------- Pipeline ----------
def make_pipeline(num, cat, params):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # handles NaNs, casts to float
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    pre = ColumnTransformer(
        [("num", num_pipe, num), ("cat", cat_pipe, cat)],
        remainder="drop",
        sparse_threshold=0.3,
    )
    xgb = XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
        reg_alpha=float(params["reg_alpha"]),
        tree_method="hist",
        n_jobs=max(1, (os.cpu_count() or 2) - 1),
        random_state=42,
        eval_metric="auc",
        enable_categorical=False,
    )
    return Pipeline([("pre", pre), ("clf", xgb)])

# ---------- Optuna objective ----------
def objective(trial, X, y, num, cat):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }
    pipe = make_pipeline(num, cat, params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()
    return auc

# ---------- Train entrypoint ----------
def train_main(data_path="data/raw/Telco-Customer-Churn.csv", model_name="xgb_churn"):
    with mlflow.start_run():
        X, y, num, cat = load_data(data_path)

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, X, y, num, cat), n_trials=25, show_progress_bar=False)
        best = study.best_params

        pipe = make_pipeline(num, cat, best)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        pipe.fit(Xtr, ytr)

        p = pipe.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p)
        mlflow.log_metric("auc", float(auc))
        mlflow.log_params(best)

        os.makedirs("artifacts", exist_ok=True)
        path = "artifacts/model.joblib"
        joblib.dump(pipe, path)
        mlflow.log_artifact(path)

        sig = infer_signature(Xte.head(5), pipe.predict_proba(Xte.head(5))[:, 1])
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",                # ok to keep despite deprecation warning
            registered_model_name=model_name,
            signature=sig,
            input_example=Xte.head(2),
        )

        print({"auc": float(auc), "best_params": best})

if __name__ == "__main__":
    train_main()
