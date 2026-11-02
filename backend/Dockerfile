# syntax=docker/dockerfile:1.7
# Backend image: API, background worker and migrations share one image.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# OCR for scanned PDFs/images (set WITH_OCR=0 to skip). LibreOffice for legacy .doc/.ppt is
# large; enable with WITH_LIBREOFFICE=1.
ARG WITH_OCR=1
ARG WITH_LIBREOFFICE=0
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libpq5 \
      $( [ "$WITH_OCR" = "1" ] && echo tesseract-ocr ) \
      $( [ "$WITH_LIBREOFFICE" = "1" ] && echo libreoffice-writer-nogui libreoffice-impress-nogui ) \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md* ./
COPY app ./app
RUN pip install ".[connectors,s3,ocr,local-embeddings]"
COPY alembic.ini ./
COPY alembic ./alembic
COPY scripts ./scripts

RUN useradd --system --uid 10001 --home /app rag && mkdir -p /data/storage && chown -R rag /data /app
USER rag

ENV STORAGE_LOCAL_ROOT=/data/storage
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 CMD curl -fsS http://localhost:8000/api/v1/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
