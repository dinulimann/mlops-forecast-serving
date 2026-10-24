# One image for the API and the MLflow server: same pinned mlflow on both ends.
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mlops mlops
COPY service service
COPY db db
ENV PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1
EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
