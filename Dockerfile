FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py model_arch.py fleurs_config.py ./

COPY models_int8_amd64/ ./models/
ENV MODEL_ROOT=/app/models

ENV PORT=8000
EXPOSE 8000

CMD uvicorn predict:app --host 0.0.0.0 --port ${PORT}
