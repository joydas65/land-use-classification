# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm AS runtime-base

ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TERRACLASS_PROJECT_ROOT=/app \
    TERRACLASS_DEVICE=cpu \
    TERRACLASS_HOST=0.0.0.0 \
    TERRACLASS_PORT=8080 \
    TERRACLASS_FAIL_ON_MODEL_ERROR=true \
    TERRACLASS_MAX_CONCURRENT_INFERENCES=1 \
    TERRACLASS_QUEUE_TIMEOUT_SECONDS=5 \
    TERRACLASS_ALLOWED_ORIGINS=https://terraclass-land-use-classification.vercel.app

WORKDIR /app

COPY requirements/serving.txt requirements/serving.txt
RUN python -m pip install --no-cache-dir pip==26.1.2 \
    && python -m pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements/serving.txt

COPY pyproject.toml README.md ./
COPY src/ src/
RUN python -m pip install --no-cache-dir --no-deps .

COPY configs/baseline_5class.json configs/baseline_5class.json
COPY configs/serving/ configs/serving/
COPY reports/inference_benchmark_2026-07-15.json reports/inference_benchmark_2026-07-15.json
COPY scripts/fetch_serving_artifact.py scripts/fetch_serving_artifact.py

RUN groupadd --system --gid 10001 terraclass \
    && useradd --system --uid 10001 --gid terraclass --home-dir /app terraclass \
    && mkdir -p /app/artifacts/serving \
    && chown -R terraclass:terraclass /app/artifacts

USER terraclass
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8080/api/v1/health/ready', timeout=2)"]

CMD ["terraclass-api"]

FROM runtime-base AS production

USER root
RUN python scripts/fetch_serving_artifact.py --project-root /app \
    && chown terraclass:terraclass /app/artifacts/serving/resnet18_group_aware_v1.pt
USER terraclass
