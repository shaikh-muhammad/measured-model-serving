FROM python:3.14.4-slim-bookworm@sha256:fc74d22ffd0d5ac395a4b7bdda75a4539758862c49ebf3005647084631e63789

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MODEL_DIR=/app/model \
    MODEL_THREADS=4

WORKDIR /app
COPY requirements-v0.lock /app/requirements-v0.lock
RUN python -m pip install --only-binary=:all: --no-deps \
    --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu
RUN python -m pip install --only-binary=:all: --no-deps -r requirements-v0.lock \
    && python -m pip check
RUN groupadd --gid 10001 serving \
    && useradd --uid 10001 --gid serving --no-create-home --home-dir /app serving
COPY src /app/src
COPY model-cache/v0 /app/model
USER 10001:10001
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
