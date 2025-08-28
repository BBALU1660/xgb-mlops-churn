from mlflow import MlflowClient
import os, sys

client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI","http://127.0.0.1:5000"))
name = "xgb_churn"

# Find versions whose RUN artifact_uri starts with mlflow-artifacts:/
goods = []
for v in client.search_model_versions(f"name='{name}'"):
    uri = client.get_run(v.run_id).info.artifact_uri
    if str(uri).startswith("mlflow-artifacts:/"):
        goods.append((int(v.version), v, uri))

if not goods:
    print("No versions with 'mlflow-artifacts:/'. Start MLflow with --serve-artifacts, retrain once, then rerun this.")
    sys.exit(1)

ver, mv, uri = sorted(goods, key=lambda x: x[0])[-1]
client.transition_model_version_stage(
    name=name, version=str(ver), stage="Production", archive_existing_versions=True
)
print(f"Promoted xgb_churn v{ver} -> Production")
print(f"Run artifact_uri: {uri}")
