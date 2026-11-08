# syntax=docker/dockerfile:1
# Build context = REPO ROOT.  docker build -f services/labs/Dockerfile .
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    PYTHONPATH=/app
WORKDIR /app

# system packages for this service only
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-ara libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 1) third-party requirements first (heavy, changes rarely -> cached layer).
#    The constraints file freezes torch/torchvision/torchaudio if the base image
#    already ships them, so pip fails loudly instead of silently replacing torch.
COPY services/labs/requirements.txt .
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio)==' > /tmp/constraints.txt || true
RUN pip install -r requirements.txt -c /tmp/constraints.txt

# 2) shared schemas (changes often, so it comes after requirements)
COPY schemas /schemas
RUN pip install /schemas

# 3) service code and tests
COPY services/labs/src /app/src
COPY services/labs/tests /app/tests

# non-root user; /models is pre-created so the named volume inherits its ownership
RUN mkdir -p /models \
    && useradd -m -u 1000 app \
    && chown -R app:app /app /models
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
