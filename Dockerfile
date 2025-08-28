FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
 && pip install --retries 10 --timeout 300 xgboost==3.0.3 \
 && pip install --retries 5 --timeout 180 -r requirements.txt

COPY src ./src

ENV MODEL_URI=models:/xgb_churn/Production

EXPOSE 8080
CMD ["uvicorn","src.serve:app","--host","0.0.0.0","--port","8080"]
