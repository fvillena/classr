FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    HOST=0.0.0.0 \
    PORT=7860 \
    MODELS_DIR=/data/models

WORKDIR /app

# Dependencias primero para aprovechar la caché de capas
COPY requirements.txt .
RUN pip install -r requirements.txt

# Usuario sin privilegios y carpeta de modelos persistente
RUN useradd --create-home --uid 1000 classr \
    && mkdir -p /data/models \
    && chown -R classr:classr /data
COPY --chown=classr:classr app.py .
COPY --chown=classr:classr ejemplos ./ejemplos

USER classr
VOLUME ["/data/models"]
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4)" || exit 1

CMD ["python", "app.py"]
