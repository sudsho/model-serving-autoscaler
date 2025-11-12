# Transformer sidecar image.
# Hosted at ghcr.io/sudsho/msa-transformer:<tag>
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# system deps for pillow + httpx tls
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libjpeg-dev zlib1g-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/

EXPOSE 8081

# Default to image task; override via TASK env.
ENV TASK=image \
    PORT=8081 \
    PREDICTOR_HOST=localhost:8080

CMD ["python", "-m", "src.transformer.server"]
