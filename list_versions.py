from mlflow import MlflowClient
import os
c = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI","http://127.0.0.1:5000"))
vers = sorted(c.search_model_versions("name='xgb_churn'"), key=lambda v: int(v.version))
for v in vers:
    print(f"v{v.version:>2}  stage={v.current_stage:<10}  source={v.source}")
