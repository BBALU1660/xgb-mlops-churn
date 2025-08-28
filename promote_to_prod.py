from mlflow import MlflowClient
import os

client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI","http://127.0.0.1:5000"))
name = "xgb_churn"

good = sorted([v for v in client.search_model_versions(f"name='{name}'")
               if str(v.source).startswith("mlflow-artifacts:/")],
              key=lambda v: int(v.version))[-1]

client.transition_model_version_stage(
    name=name,
    version=good.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"Promoted v{good.version} -> Production")
