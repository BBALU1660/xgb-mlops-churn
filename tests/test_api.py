def test_predict_endpoint(monkeypatch):
    import numpy as np
    import mlflow.sklearn as mls
    from importlib import import_module
    from fastapi.testclient import TestClient

    # Dummy model that mimics predict_proba
    class DummyModel:
        def predict_proba(self, X):
            # always return 0.42 for positive class
            return np.column_stack([np.zeros(len(X)), np.full(len(X), 0.42)])

    # Patch mlflow loader BEFORE importing the app
    monkeypatch.setattr(mls, "load_model", lambda uri: DummyModel())

    serve = import_module("src.serve")  # imports after patch; no real MLflow call
    client = TestClient(serve.app)

    payload = {
        "rows": [{
            "data": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
                "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
                "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
                "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
            }
        }]
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "probs" in body and isinstance(body["probs"][0], float)
