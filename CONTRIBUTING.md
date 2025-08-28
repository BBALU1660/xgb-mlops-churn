# Contributing

## Dev setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install pytest black pre-commit httpx
Run tests
powershell
Copy code
pytest -q
Format code
powershell
Copy code
black src tests
Start locally (no Docker)
powershell
Copy code
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
uvicorn src.serve:app --host 127.0.0.1 --port 8000 --reload
Commit style
Small, focused commits

Messages: feat: ..., fix: ..., chore: ..., docs: ..., test: ...
