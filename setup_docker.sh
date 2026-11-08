#!/usr/bin/env bash
# MICA: generates the Docker scaffolding for all 5 services + compose + CI.
# Run from the ROOT of the repo:   bash setup_docker.sh
# Existing files are never overwritten (use FORCE=1 to overwrite).
set -euo pipefail

SERVICES="api asr labs xray llm"
PYTORCH_BASE="pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"

write() {
  local path="$1"
  if [ -e "$path" ] && [ -z "${FORCE:-}" ]; then
    echo "skip (exists): $path"
    cat > /dev/null
  else
    mkdir -p "$(dirname "$path")"
    cat > "$path"
    echo "wrote:         $path"
  fi
}

# ---------------------------------------------------------------- Dockerfiles
gen_dockerfile() {  # $1=service  $2=default base image  $3=apt packages (may be empty)
  local svc="$1" base="$2" apt="$3"
  {
    echo "# syntax=docker/dockerfile:1"
    echo "# Build context = REPO ROOT.  docker build -f services/$svc/Dockerfile ."
    echo "ARG BASE_IMAGE=$base"
    cat <<'EOF'
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    PYTHONPATH=/app
WORKDIR /app

EOF
    if [ -n "$apt" ]; then
      printf '# system packages for this service only\n'
      printf 'RUN apt-get update && apt-get install -y --no-install-recommends %s \\\n    && rm -rf /var/lib/apt/lists/*\n\n' "$apt"
    fi
    cat <<EOF
# 1) third-party requirements first (heavy, changes rarely -> cached layer).
#    The constraints file freezes torch/torchvision/torchaudio if the base image
#    already ships them, so pip fails loudly instead of silently replacing torch.
COPY services/$svc/requirements.txt .
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio)==' > /tmp/constraints.txt || true
RUN pip install -r requirements.txt -c /tmp/constraints.txt

# 2) shared schemas (changes often, so it comes after requirements)
COPY schemas /schemas
RUN pip install /schemas

# 3) service code and tests
COPY services/$svc/src /app/src
COPY services/$svc/tests /app/tests

# non-root user; /models is pre-created so the named volume inherits its ownership
RUN mkdir -p /models \\
    && useradd -m -u 1000 app \\
    && chown -R app:app /app /models
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \\
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
  } | write "services/$svc/Dockerfile"
}

gen_dockerfile api  "python:3.11-slim" ""
gen_dockerfile llm  "python:3.11-slim" ""
gen_dockerfile labs "python:3.11-slim" "tesseract-ocr tesseract-ocr-ara libglib2.0-0"
gen_dockerfile asr  "$PYTORCH_BASE"    "ffmpeg"
gen_dockerfile xray "$PYTORCH_BASE"    "libglib2.0-0"

# ------------------------------------------------------ requirements (stubs)
# Versions are intentionally loose in the stubs: after the first successful build,
# each owner pins the exact versions they tested (pip freeze inside the container).
write services/api/requirements.txt <<'EOF'
# OWNER: backend. Pin exact versions after the first successful build.
fastapi
uvicorn
httpx
pytest
# add: sqlalchemy, psycopg[binary], redis, celery or rq, python-jose, python-multipart ...
EOF

write services/llm/requirements.txt <<'EOF'
# OWNER: LLM/leader. Pin exact versions after the first successful build.
fastapi
uvicorn
httpx
pytest
openai
rapidfuzz
EOF

write services/labs/requirements.txt <<'EOF'
# OWNER: DIP. Pin exact versions after the first successful build.
fastapi
uvicorn
httpx
pytest
# add: opencv-python-headless (NOT opencv-python), pytesseract, easyocr or paddleocr,
#      pymupdf, rapidfuzz, numpy ...
EOF

write services/asr/requirements.txt <<'EOF'
# OWNER: audio. Base image already has torch: do NOT list torch/torchaudio here.
# Pin exact versions after the first successful build.
fastapi
uvicorn
httpx
pytest
# add: faster-whisper, pyannote.audio, soundfile, jiwer, librosa ...
# After install run `pip list | grep -i torch` and make sure torch was not replaced.
EOF

write services/xray/requirements.txt <<'EOF'
# OWNER: x-ray. Base image already has torch: do NOT list torch/torchvision here.
# Pin exact versions after the first successful build.
fastapi
uvicorn
httpx
pytest
# add: torchxrayvision, grad-cam, pydicom, opencv-python-headless, scikit-learn ...
EOF

# ------------------------------------------------------- service stubs + tests
for svc in $SERVICES; do
  upper=$(echo "$svc" | tr '[:lower:]' '[:upper:]')
  write "services/$svc/src/__init__.py" </dev/null
  write "services/$svc/src/main.py" <<EOF
import os

from fastapi import FastAPI

SERVICE = "$svc"
app = FastAPI(title=f"mica-{SERVICE}")


@app.get("/health")
def health():
    return {
        "service": SERVICE,
        "status": "ok",
        "fake": os.getenv("${upper}_FAKE", "1") == "1",
    }


# TODO(owner): replace the stub with the real endpoint, e.g. POST /run.
# Inputs and outputs must validate against mica_schemas (contract test).
EOF
  write "services/$svc/tests/test_smoke.py" <<'EOF'
def test_schemas_installed():
    # fails if `pip install /schemas` did not work inside the image
    import mica_schemas  # noqa: F401


def test_health():
    from fastapi.testclient import TestClient

    from src.main import app

    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
EOF
done

# ------------------------------------------------------------ docker-compose
write docker-compose.yml <<'EOF'
# Run only what you need:   docker compose up api redis postgres
# GPU services (asr, xray) and ollama are behind profiles, so a plain
# `docker compose up` works on a machine without a GPU.
# Naming a service explicitly starts it:   docker compose up asr
name: mica

x-common: &common
  env_file: .env

x-gpu: &gpu
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]

services:
  redis:
    image: redis:7
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10

  postgres:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?copy .env.example to .env and set POSTGRES_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 3s
      retries: 10

  api:
    <<: *common
    build: {context: ., dockerfile: services/api/Dockerfile}
    ports: ["8000:8000"]
    depends_on:
      redis: {condition: service_healthy}
      postgres: {condition: service_healthy}

  llm:
    <<: *common
    build: {context: ., dockerfile: services/llm/Dockerfile}
    ports: ["8004:8000"]

  labs:
    <<: *common
    build: {context: ., dockerfile: services/labs/Dockerfile}
    ports: ["8002:8000"]

  asr:
    <<: *common
    profiles: ["gpu"]
    build: {context: ., dockerfile: services/asr/Dockerfile}
    volumes: [hf_models:/models]
    ports: ["8001:8000"]
    deploy: *gpu

  xray:
    <<: *common
    profiles: ["gpu"]
    build: {context: ., dockerfile: services/xray/Dockerfile}
    volumes: [hf_models:/models]
    ports: ["8003:8000"]
    deploy: *gpu

  ollama:
    profiles: ["local-llm"]
    image: ollama/ollama
    volumes: [ollama:/root/.ollama]
    ports: ["11434:11434"]
    deploy: *gpu   # no GPU? delete this line and Ollama runs on CPU (slow)

volumes:
  hf_models:
  pgdata:
  ollama:
EOF

# ----------------------------------------------------------------- env files
write .env.example <<'EOF'
# cp .env.example .env     (.env is git-ignored: NEVER commit real keys)

POSTGRES_PASSWORD=change_me
HF_TOKEN=

# --- LLM (OpenAI-compatible endpoint) ---
# Synthetic / de-identified data only when this points to an external API.
LLM_BASE_URL=http://ollama:11434/v1
#   external example (verify the URL in the provider docs): https://api.groq.com/openai/v1
#   Ollama on the Windows host (not in compose): http://host.docker.internal:11434/v1
LLM_MODEL=
LLM_API_KEY=ollama

# --- fake mode: 1 = return the fixture instead of running the real model ---
API_FAKE=1
ASR_FAKE=1
LABS_FAKE=1
XRAY_FAKE=1
LLM_FAKE=1
EOF

write .dockerignore <<'EOF'
.git
.github
.env
.venv
**/__pycache__
**/.pytest_cache
**/*.pyc
data/
models/
*.wav
*.mp3
*.m4a
*.dcm
EOF

# --------------------------------------------------------------------- CI
write .github/workflows/ci.yml <<'EOF'
name: ci
on:
  pull_request:
  push:
    branches: [main]

jobs:
  build-test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        service: [api, asr, labs, xray, llm]
    steps:
      - uses: actions/checkout@v4
      - name: Build image
        run: docker build -f services/${{ matrix.service }}/Dockerfile -t mica-${{ matrix.service }}:ci .
      - name: Run tests inside the image
        run: docker run --rm mica-${{ matrix.service }}:ci python -m pytest -q
      # Pushing images to Docker Hub on main is the backend owner's job
      # (needs DOCKERHUB_USERNAME / DOCKERHUB_TOKEN repository secrets).
EOF

# .gitignore: make sure secrets/data are ignored (append only what is missing)
touch .gitignore
for line in ".env" ".venv/" "__pycache__/" "data/" "models/" "*.wav" "*.mp3"; do
  grep -qxF "$line" .gitignore || echo "$line" >> .gitignore
done

echo
echo "Done. Next:"
echo "  cp .env.example .env   # then edit it"
echo "  docker compose build llm"
echo "  docker compose run --rm llm python -m pytest -q"
