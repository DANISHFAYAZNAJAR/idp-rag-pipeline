FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
COPY app ./app
COPY worker ./worker
COPY mcp_server ./mcp_server
COPY ui ./ui
COPY alembic ./alembic
COPY alembic.ini ./
COPY scripts ./scripts

RUN uv sync --frozen --no-dev --extra ui \
    && chmod +x scripts/docker-entrypoint.sh scripts/start_api.sh scripts/start_ui.sh scripts/start_worker.sh

RUN mkdir -p /data/chroma_store /data/uploads /data/logs

VOLUME ["/data/chroma_store", "/data/uploads", "/data/logs"]

EXPOSE 8080 8501 8001

ENTRYPOINT ["scripts/docker-entrypoint.sh"]
CMD ["api"]
