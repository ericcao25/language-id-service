FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "\
from transformers import AutoFeatureExtractor, AutoConfig; \
AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-xls-r-300m').save_pretrained('/app/base_model_config'); \
AutoConfig.from_pretrained('facebook/wav2vec2-xls-r-300m').save_pretrained('/app/base_model_config')"

COPY predict.py model_arch.py fleurs_config.py ./

ENV HF_HUB_OFFLINE=1
ENV PORT=8000
EXPOSE 8000

CMD uvicorn predict:app --host 0.0.0.0 --port ${PORT}
