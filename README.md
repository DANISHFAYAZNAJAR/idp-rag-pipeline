# IDP RAG

Document Q&A with hybrid retrieval (BM25 + vector search), reranking, and a Streamlit UI.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- PostgreSQL and Redis running locally
- `.env` configured (copy from `.env.example`)

## Setup

```bash
cd /path/to/RAG_IDP
uv sync --extra ui
cp .env.example .env   # then edit with your keys and DB URLs
uv run alembic upgrade head
```

## Run locally

Use **3 terminals**. Start API and worker before the UI.

### 1. API server

```bash
./scripts/start_api.sh
```

- http://localhost:8080
- Docs: http://localhost:8080/docs

### 2. Worker (document ingestion)

```bash
./scripts/start_worker.sh
```

### 3. Streamlit UI

```bash
./scripts/start_ui.sh
```

- http://localhost:8501

## Ports (`.env`)

| Variable | Default |
|----------|---------|
| `API_PORT` | 8080 |
| `UI_PORT` | 8501 |
| `API_PUBLIC_URL` | http://localhost:8080 |

## Docker (optional)

```bash
docker compose build
docker compose up -d
```

See `docker-compose.yml` for service ports and env overrides.
