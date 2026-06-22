# IDP RAG System — Full Implementation Blueprint
> **Goal:** Intelligent Document Processing (IDP) system with RAG pipeline.  
> **Approach:** Build FastAPI core + RAG first. Docker, Prometheus, Grafana added later when deploying to server.  
> **Use this document in Cursor:** Feed it to Cursor AI as context. Implement phase by phase in order.

---

## Tech Stack (Final)

| Layer | Technology | Notes |
|---|---|---|
| API framework | FastAPI + uvicorn | Async throughout |
| Task queue | Celery + Redis | Background ingestion jobs |
| Vector store | ChromaDB | Local persistent mode |
| Relational DB | PostgreSQL (asyncpg) | Docs, chunks, users, sessions |
| LLM | OpenAI GPT-4o | Streaming responses |
| Embeddings | OpenAI text-embedding-3-small | Batched |
| RAG orchestration | LangChain + LangGraph | Graph-based retrieval pipeline |
| PDF parsing | pdfplumber + unstructured | Text + tables |
| MCP server | FastMCP (Python) | Exposes RAG as MCP tools |
| Tracing | W&B Weave | LLM traces + evals |
| Logging | structlog → JSONL | ./logs/app.jsonl |
| Auth | python-jose + passlib | JWT |
| File storage | Local disk | ./uploads/ (swap to S3 later) |
| Metrics | prometheus-fastapi-instrumentator | Off by default, enable on server |

---

## Project Structure

```
idp-rag/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app factory
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py                # Shared dependencies (JWT, DB session)
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── auth.py            # /auth/register, /auth/login
│   │       ├── documents.py       # Upload, list, status, delete
│   │       ├── query.py           # Q&A, streaming, history
│   │       ├── entities.py        # NER results per document
│   │       └── health.py          # Liveness + readiness
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Pydantic settings
│   │   ├── logging.py             # Structlog JSONL setup
│   │   ├── exceptions.py          # Custom HTTP exceptions
│   │   └── security.py            # JWT encode/decode, password hash
│   ├── db/
│   │   ├── __init__.py
│   │   ├── base.py                # SQLAlchemy async engine + session
│   │   └── models.py              # ORM models
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py          # pdfplumber + unstructured
│   │   ├── chunker.py             # Semantic + recursive splitter
│   │   ├── embedder.py            # OpenAI embeddings wrapper
│   │   ├── enricher.py            # Doc classifier + NER
│   │   ├── storage.py             # LocalStorageService (S3-swappable)
│   │   └── pipeline.py            # Celery task chain
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── chroma_store.py        # ChromaDB client wrapper
│   │   ├── bm25_retriever.py      # BM25 over PostgreSQL chunks
│   │   ├── hybrid_search.py       # RRF fusion
│   │   ├── query_rewriter.py      # HyDE + multi-query expansion
│   │   ├── reranker.py            # Cross-encoder reranker
│   │   └── graph.py               # LangGraph RAG StateGraph
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── documents.py
│   │   └── query.py
│   └── services/
│       ├── __init__.py
│       ├── document_service.py    # Business logic for docs
│       └── query_service.py       # Orchestrates RAG graph
├── mcp_server/
│   ├── __init__.py
│   └── server.py                  # FastMCP server exposing RAG tools
├── worker/
│   ├── __init__.py
│   └── celery_app.py
├── alembic/
│   ├── env.py
│   └── versions/
├── tests/
│   ├── conftest.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_api.py
├── logs/                          # JSONL logs written here
├── uploads/                       # Raw PDFs stored here
├── chroma_store/                  # ChromaDB persistence
├── .env
├── .env.example
├── alembic.ini
├── pyproject.toml
└── README.md
```

---

## Environment Variables

### `.env.example`
```env
# ── OpenAI ──────────────────────────────────────────────
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o

# ── W&B Weave ────────────────────────────────────────────
WANDB_API_KEY=...
WANDB_PROJECT=idp-rag
WANDB_ENTITY=your-wandb-username

# ── Database ─────────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/idprag
DATABASE_SYNC_URL=postgresql://user:pass@localhost:5432/idprag   # for Alembic

# ── Redis (Celery broker) ────────────────────────────────
REDIS_URL=redis://localhost:6379/0

# ── ChromaDB ─────────────────────────────────────────────
CHROMA_PERSIST_DIR=./chroma_store
CHROMA_COLLECTION_NAME=idp_chunks

# ── Local Storage ────────────────────────────────────────
STORAGE_BACKEND=local
LOCAL_STORAGE_PATH=./uploads

# ── App ──────────────────────────────────────────────────
SECRET_KEY=change-me-to-something-long-and-random
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
ENVIRONMENT=development

# ── Logging ──────────────────────────────────────────────
LOG_LEVEL=INFO
LOG_FILE=./logs/app.jsonl
LOG_MAX_BYTES=10485760       # 10 MB before rotation
LOG_BACKUP_COUNT=5

# ── Rate limiting ────────────────────────────────────────
RATE_LIMIT_UPLOADS_PER_HOUR=10
RATE_LIMIT_QUERIES_PER_HOUR=100

# ── Feature flags ────────────────────────────────────────
ENABLE_METRICS=false          # flip true on server
ENABLE_RERANKER=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# ── MCP Server ───────────────────────────────────────────
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8001

# ── Ingestion ────────────────────────────────────────────
CHUNK_SIZE=800
CHUNK_OVERLAP=100
MAX_CHUNKS_PER_DOC=2000
EMBEDDING_BATCH_SIZE=50
RETRIEVAL_TOP_K=20
RERANKER_TOP_N=5
CONFIDENCE_THRESHOLD=0.35
```

---

## Dependencies

### `pyproject.toml`
```toml
[project]
name = "idp-rag"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # API
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.29.0",
    "python-multipart>=0.0.9",
    "slowapi>=0.1.9",

    # Database
    "sqlalchemy[asyncio]>=2.0.30",
    "asyncpg>=0.29.0",
    "alembic>=1.13.1",
    "psycopg2-binary>=2.9.9",   # Alembic sync driver

    # Auth
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",

    # Task queue
    "celery[redis]>=5.4.0",
    "redis>=5.0.4",

    # LLM / LangChain
    "langchain>=0.2.0",
    "langchain-openai>=0.1.0",
    "langchain-chroma>=0.1.0",
    "langchain-community>=0.2.0",
    "langgraph>=0.1.0",
    "openai>=1.30.0",

    # Vector store
    "chromadb>=0.5.0",

    # PDF parsing
    "pdfplumber>=0.11.0",
    "unstructured[pdf]>=0.14.0",
    "pytesseract>=0.3.10",       # OCR fallback
    "Pillow>=10.3.0",

    # NLP
    "spacy>=3.7.4",
    "rank-bm25>=0.2.2",
    "sentence-transformers>=3.0.0",  # cross-encoder reranker

    # Observability
    "wandb>=0.17.0",
    "weave>=0.50.0",
    "structlog>=24.2.0",
    "prometheus-fastapi-instrumentator>=7.0.0",

    # MCP
    "fastmcp>=0.1.0",

    # Utilities
    "pydantic>=2.7.0",
    "pydantic-settings>=2.3.0",
    "httpx>=0.27.0",
    "aiofiles>=23.2.1",
    "python-dotenv>=1.0.1",
    "tenacity>=8.3.0",           # retry logic
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2.0",
    "pytest-asyncio>=0.23.7",
    "pytest-cov>=5.0.0",
    "httpx>=0.27.0",
    "factory-boy>=3.3.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
]
```

---

## Phase 1 — Core Config & Logging

### `app/core/config.py`
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o"

    # W&B
    wandb_api_key: str = ""
    wandb_project: str = "idp-rag"
    wandb_entity: str = ""

    # Database
    database_url: str
    database_sync_url: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_store"
    chroma_collection_name: str = "idp_chunks"

    # Storage
    storage_backend: str = "local"
    local_storage_path: str = "./uploads"

    # Auth
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    # App
    environment: str = "development"
    log_level: str = "INFO"
    log_file: str = "./logs/app.jsonl"
    log_max_bytes: int = 10_485_760
    log_backup_count: int = 5

    # Rate limiting
    rate_limit_uploads_per_hour: int = 10
    rate_limit_queries_per_hour: int = 100

    # Feature flags
    enable_metrics: bool = False
    enable_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # MCP
    mcp_server_host: str = "0.0.0.0"
    mcp_server_port: int = 8001

    # Ingestion
    chunk_size: int = 800
    chunk_overlap: int = 100
    max_chunks_per_doc: int = 2000
    embedding_batch_size: int = 50
    retrieval_top_k: int = 20
    reranker_top_n: int = 5
    confidence_threshold: float = 0.35


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
```

### `app/core/logging.py`
```python
import logging
import logging.handlers
import sys
from pathlib import Path

import structlog

from app.core.config import settings


def setup_logging() -> None:
    Path(settings.log_file).parent.mkdir(parents=True, exist_ok=True)

    # Rotating JSONL file handler
    file_handler = logging.handlers.RotatingFileHandler(
        filename=settings.log_file,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )

    # Console handler (plain text in dev, JSON in prod)
    console_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.log_level.upper()),
        handlers=[file_handler, console_handler],
    )

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.environment == "development":
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),  # always JSON to file
        foreign_pre_chain=shared_processors,
    )
    file_handler.setFormatter(formatter)
```

### `app/core/exceptions.py`
```python
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class DocumentNotFoundError(HTTPException):
    def __init__(self, document_id: str):
        super().__init__(
            status_code=404,
            detail={"code": "DOCUMENT_NOT_FOUND", "document_id": document_id}
        )


class DocumentProcessingError(HTTPException):
    def __init__(self, message: str):
        super().__init__(
            status_code=422,
            detail={"code": "PROCESSING_ERROR", "message": message}
        )


class InsufficientContextError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=200,   # still 200 — answer exists, just low confidence
            detail={"code": "INSUFFICIENT_CONTEXT",
                    "message": "Not enough relevant content found in the document."}
        )


class StorageError(HTTPException):
    def __init__(self, message: str):
        super().__init__(
            status_code=500,
            detail={"code": "STORAGE_ERROR", "message": message}
        )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )
```

### `app/core/security.py`
```python
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str | Any) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    payload = {"sub": str(subject), "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_token(token: str) -> str:
    """Returns user_id from token or raises JWTError."""
    payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    return payload["sub"]
```

---

## Phase 2 — Database Models

### `app/db/base.py`
```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

engine = create_async_engine(
    settings.database_url,
    echo=settings.environment == "development",
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

### `app/db/models.py`
```python
import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey,
    Integer, JSON, String, Text, Enum, Boolean
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base import Base


def utcnow():
    return datetime.now(timezone.utc)


class DocumentStatus(str, PyEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    ENRICHING = "enriching"
    READY = "ready"
    FAILED = "failed"


class DocumentType(str, PyEnum):
    INVOICE = "invoice"
    CONTRACT = "contract"
    REPORT = "report"
    RESEARCH = "research"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("QuerySession", back_populates="user", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size_bytes = Column(Integer)
    page_count = Column(Integer)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.QUEUED, index=True)
    progress = Column(Float, default=0.0)           # 0.0 → 1.0
    doc_type = Column(Enum(DocumentType), default=DocumentType.UNKNOWN)
    doc_metadata = Column(JSON, default=dict)        # title, author, sections, etc.
    error_message = Column(Text)
    celery_task_id = Column(String(255))
    chroma_collection_id = Column(String(255))       # user-scoped Chroma collection
    created_at = Column(DateTime(timezone=True), default=utcnow)
    processed_at = Column(DateTime(timezone=True))

    user = relationship("User", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    entities = relationship("Entity", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    page_number = Column(Integer)
    section_heading = Column(String(500))
    chunk_type = Column(String(50), default="text")  # text | table | figure_caption
    chroma_id = Column(String(255), index=True)      # ID in ChromaDB
    token_count = Column(Integer)
    chunk_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    document = relationship("Document", back_populates="chunks")


class Entity(Base):
    __tablename__ = "entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)
    entity_type = Column(String(100))   # PERSON, ORG, DATE, MONEY, LOC, etc.
    entity_text = Column(String(1000))
    page_number = Column(Integer)
    confidence = Column(Float)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    document = relationship("Document", back_populates="entities")


class QuerySession(Base):
    __tablename__ = "query_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    document_ids = Column(JSON, default=list)        # can query across multiple docs
    created_at = Column(DateTime(timezone=True), default=utcnow)

    user = relationship("User", back_populates="sessions")
    turns = relationship("QueryTurn", back_populates="session", cascade="all, delete-orphan")


class QueryTurn(Base):
    __tablename__ = "query_turns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("query_sessions.id"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text)
    citations = Column(JSON, default=list)           # [{chunk_id, page, snippet}]
    confidence = Column(Float)
    retrieved_chunks = Column(Integer)
    latency_ms = Column(Integer)
    weave_trace_id = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=utcnow)

    session = relationship("QuerySession", back_populates="turns")
```

---

## Phase 3 — Storage Service

### `app/ingestion/storage.py`
```python
import uuid
from pathlib import Path
from abc import ABC, abstractmethod

import aiofiles
import structlog

from app.core.config import settings

log = structlog.get_logger()


class BaseStorageService(ABC):
    @abstractmethod
    async def save(self, file_bytes: bytes, user_id: str, filename: str) -> str:
        """Save file and return its storage path."""

    @abstractmethod
    async def get(self, path: str) -> bytes:
        """Retrieve file bytes by path."""

    @abstractmethod
    async def delete(self, path: str) -> None:
        """Delete file by path."""

    @abstractmethod
    def get_absolute_path(self, path: str) -> Path:
        """Return absolute Path for local access (PDF parsing needs this)."""


class LocalStorageService(BaseStorageService):
    def __init__(self, base_path: str = settings.local_storage_path):
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)

    async def save(self, file_bytes: bytes, user_id: str, filename: str) -> str:
        safe_name = f"{uuid.uuid4()}_{Path(filename).name}"
        user_dir = self.base / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        dest = user_dir / safe_name
        async with aiofiles.open(dest, "wb") as f:
            await f.write(file_bytes)
        rel_path = str(dest.relative_to(self.base))
        log.info("storage.saved", path=rel_path, size_bytes=len(file_bytes))
        return rel_path

    async def get(self, path: str) -> bytes:
        full = self.base / path
        async with aiofiles.open(full, "rb") as f:
            return await f.read()

    async def delete(self, path: str) -> None:
        full = self.base / path
        if full.exists():
            full.unlink()
            log.info("storage.deleted", path=path)

    def get_absolute_path(self, path: str) -> Path:
        return self.base / path


def get_storage_service() -> BaseStorageService:
    """Factory — swap STORAGE_BACKEND=s3 here later."""
    if settings.storage_backend == "local":
        return LocalStorageService()
    raise ValueError(f"Unknown storage backend: {settings.storage_backend}")


storage_service = get_storage_service()
```

---

## Phase 4 — PDF Ingestion Pipeline

### `app/ingestion/pdf_parser.py`
```python
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
import structlog
from unstructured.partition.pdf import partition_pdf

log = structlog.get_logger()


@dataclass
class ParsedPage:
    page_number: int
    text: str
    tables: list[str] = field(default_factory=list)   # markdown-formatted tables
    has_images: bool = False


@dataclass
class ParsedDocument:
    pages: list[ParsedPage]
    total_pages: int
    metadata: dict

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


def parse_pdf(file_path: Path) -> ParsedDocument:
    """
    Primary: pdfplumber for text + tables.
    Fallback: unstructured for scanned/complex PDFs.
    """
    log.info("pdf_parser.start", path=str(file_path))
    pages: list[ParsedPage] = []
    metadata: dict = {}

    try:
        with pdfplumber.open(file_path) as pdf:
            metadata = {
                "title": pdf.metadata.get("Title", ""),
                "author": pdf.metadata.get("Author", ""),
                "creator": pdf.metadata.get("Creator", ""),
                "total_pages": len(pdf.pages),
            }
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                tables_md = []
                for table in page.extract_tables():
                    if table:
                        rows = [" | ".join(str(c or "") for c in row) for row in table]
                        tables_md.append("\n".join(rows))

                # If page has very little text, attempt unstructured OCR
                if len(text.strip()) < 50:
                    text = _fallback_unstructured(file_path, i)

                pages.append(ParsedPage(
                    page_number=i,
                    text=text.strip(),
                    tables=tables_md,
                    has_images=len(page.images) > 0,
                ))

    except Exception as e:
        log.error("pdf_parser.pdfplumber_failed", error=str(e), path=str(file_path))
        # Full fallback to unstructured
        pages = _full_unstructured_parse(file_path)

    log.info("pdf_parser.complete", total_pages=len(pages))
    return ParsedDocument(pages=pages, total_pages=len(pages), metadata=metadata)


def _fallback_unstructured(file_path: Path, page_number: int) -> str:
    try:
        elements = partition_pdf(
            filename=str(file_path),
            strategy="hi_res",
            include_page_breaks=True,
        )
        page_texts = [e.text for e in elements
                      if hasattr(e, "metadata")
                      and getattr(e.metadata, "page_number", None) == page_number
                      and e.text]
        return "\n".join(page_texts)
    except Exception as e:
        log.warning("pdf_parser.unstructured_fallback_failed", page=page_number, error=str(e))
        return ""


def _full_unstructured_parse(file_path: Path) -> list[ParsedPage]:
    elements = partition_pdf(filename=str(file_path), strategy="fast")
    page_map: dict[int, list[str]] = {}
    for el in elements:
        pn = getattr(getattr(el, "metadata", None), "page_number", 1) or 1
        page_map.setdefault(pn, []).append(el.text or "")
    return [
        ParsedPage(page_number=pn, text="\n".join(texts))
        for pn, texts in sorted(page_map.items())
    ]
```

### `app/ingestion/chunker.py`
```python
from dataclasses import dataclass

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.ingestion.pdf_parser import ParsedDocument

log = structlog.get_logger()


@dataclass
class Chunk:
    text: str
    page_number: int
    chunk_index: int
    section_heading: str
    chunk_type: str      # "text" | "table"
    token_count: int
    metadata: dict


def chunk_document(parsed: ParsedDocument) -> list[Chunk]:
    """
    Strategy:
    - Tables → stored as individual chunks (one table = one chunk)
    - Narrative text → SemanticChunker for long docs, RecursiveCharacterTextSplitter as fallback
    """
    chunks: list[Chunk] = []
    index = 0

    # Extract table chunks first
    for page in parsed.pages:
        for table_md in page.tables:
            if table_md.strip():
                chunks.append(Chunk(
                    text=f"[TABLE]\n{table_md}",
                    page_number=page.page_number,
                    chunk_index=index,
                    section_heading="",
                    chunk_type="table",
                    token_count=_estimate_tokens(table_md),
                    metadata={"page": page.page_number, "type": "table"},
                ))
                index += 1

    # Chunk narrative text
    text_chunks = _chunk_text(parsed, start_index=index)
    chunks.extend(text_chunks)

    log.info("chunker.complete", total_chunks=len(chunks))
    return chunks[:settings.max_chunks_per_doc]


def _chunk_text(parsed: ParsedDocument, start_index: int) -> list[Chunk]:
    """Use SemanticChunker if doc > 10 pages, else RecursiveCharacterTextSplitter."""
    chunks: list[Chunk] = []

    # Build page-annotated text segments
    segments: list[tuple[int, str]] = [
        (p.page_number, p.text) for p in parsed.pages if p.text.strip()
    ]

    if parsed.total_pages > 10:
        log.info("chunker.strategy", strategy="semantic")
        chunks = _semantic_chunk(segments, start_index)
    else:
        log.info("chunker.strategy", strategy="recursive")
        chunks = _recursive_chunk(segments, start_index)

    return chunks


def _semantic_chunk(segments: list[tuple[int, str]], start_index: int) -> list[Chunk]:
    embeddings = OpenAIEmbeddings(model=settings.openai_embedding_model)
    splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

    chunks: list[Chunk] = []
    idx = start_index

    for page_num, text in segments:
        try:
            docs = splitter.create_documents([text])
            for doc in docs:
                chunks.append(Chunk(
                    text=doc.page_content,
                    page_number=page_num,
                    chunk_index=idx,
                    section_heading=_detect_heading(doc.page_content),
                    chunk_type="text",
                    token_count=_estimate_tokens(doc.page_content),
                    metadata={"page": page_num, "type": "text"},
                ))
                idx += 1
        except Exception as e:
            log.warning("chunker.semantic_failed", page=page_num, error=str(e))
            # Fallback for this page
            for c in _recursive_chunk([(page_num, text)], idx):
                chunks.append(c)
                idx += 1

    return chunks


def _recursive_chunk(segments: list[tuple[int, str]], start_index: int) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    idx = start_index

    for page_num, text in segments:
        texts = splitter.split_text(text)
        for t in texts:
            if t.strip():
                chunks.append(Chunk(
                    text=t,
                    page_number=page_num,
                    chunk_index=idx,
                    section_heading=_detect_heading(t),
                    chunk_type="text",
                    token_count=_estimate_tokens(t),
                    metadata={"page": page_num, "type": "text"},
                ))
                idx += 1

    return chunks


def _detect_heading(text: str) -> str:
    """Heuristic: first line is heading if short and title-cased."""
    first_line = text.strip().split("\n")[0].strip()
    if len(first_line) < 100 and first_line.istitle():
        return first_line
    return ""


def _estimate_tokens(text: str) -> int:
    return len(text) // 4  # rough approximation
```

### `app/ingestion/embedder.py`
```python
import asyncio
from typing import Any

import structlog
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.ingestion.chunker import Chunk

log = structlog.get_logger()


class EmbedderService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )

    async def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Batch embed chunks with retry on rate limit."""
        texts = [c.text for c in chunks]
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), settings.embedding_batch_size):
            batch = texts[i: i + settings.embedding_batch_size]
            log.info("embedder.batch", batch_num=i // settings.embedding_batch_size,
                     batch_size=len(batch))
            try:
                vectors = await asyncio.get_event_loop().run_in_executor(
                    None, self.embeddings.embed_documents, batch
                )
                all_embeddings.extend(vectors)
            except Exception as e:
                log.error("embedder.batch_failed", error=str(e), batch_start=i)
                raise

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        return await asyncio.get_event_loop().run_in_executor(
            None, self.embeddings.embed_query, query
        )


embedder_service = EmbedderService()
```

### `app/ingestion/enricher.py`
```python
import json
from typing import Any

import structlog
from openai import AsyncOpenAI

from app.core.config import settings
from app.db.models import DocumentType

log = structlog.get_logger()

client = AsyncOpenAI(api_key=settings.openai_api_key)

CLASSIFY_PROMPT = """You are a document classifier. Given the first 2000 characters of a document, 
classify it into one of: invoice, contract, report, research, manual, unknown.
Also extract: title, author (if visible), key_dates (list), summary (2 sentences).

Respond ONLY with valid JSON:
{
  "doc_type": "...",
  "title": "...",
  "author": "...",
  "key_dates": [],
  "summary": "..."
}"""

NER_PROMPT = """Extract named entities from this text. Return ONLY valid JSON:
{
  "entities": [
    {"type": "PERSON|ORG|DATE|MONEY|LOCATION|PRODUCT|LAW", "text": "...", "confidence": 0.0-1.0}
  ]
}"""


async def classify_document(text_sample: str) -> dict[str, Any]:
    """Classify document type and extract top-level metadata."""
    try:
        response = await client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user", "content": text_sample[:2000]},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        doc_type_str = result.get("doc_type", "unknown").lower()
        result["doc_type"] = DocumentType(doc_type_str) if doc_type_str in DocumentType._value2member_map_ else DocumentType.UNKNOWN
        return result
    except Exception as e:
        log.error("enricher.classify_failed", error=str(e))
        return {"doc_type": DocumentType.UNKNOWN}


async def extract_entities(text_chunk: str, page_number: int) -> list[dict[str, Any]]:
    """Run NER on a text chunk. Called per-chunk during ingestion."""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",   # cheaper model for NER
            messages=[
                {"role": "system", "content": NER_PROMPT},
                {"role": "user", "content": text_chunk[:1500]},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        entities = result.get("entities", [])
        for e in entities:
            e["page_number"] = page_number
        return entities
    except Exception as e:
        log.warning("enricher.ner_failed", page=page_number, error=str(e))
        return []
```

### `app/ingestion/pipeline.py`
```python
"""
Celery task chain for full document ingestion.
Each step updates document.progress in PostgreSQL.
"""
import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog
from celery import chain, shared_task
from sqlalchemy import update

from app.core.config import settings
from app.db.base import AsyncSessionLocal
from app.db.models import Chunk, Document, DocumentStatus, Entity
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import embedder_service
from app.ingestion.enricher import classify_document, extract_entities
from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.storage import storage_service
from app.retrieval.chroma_store import chroma_service

log = structlog.get_logger()


def run_async(coro):
    """Run async coroutine from a sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _update_status(doc_id: str, status: DocumentStatus,
                          progress: float = None, error: str = None) -> None:
    async with AsyncSessionLocal() as session:
        values = {"status": status}
        if progress is not None:
            values["progress"] = progress
        if error:
            values["error_message"] = error
        if status == DocumentStatus.READY:
            values["processed_at"] = datetime.now(timezone.utc)
        await session.execute(
            update(Document).where(Document.id == uuid.UUID(doc_id)).values(**values)
        )
        await session.commit()


@shared_task(bind=True, max_retries=3, default_retry_delay=30)
def task_ingest_document(self, document_id: str) -> dict:
    """
    Full ingestion pipeline as a single Celery task with progress updates.
    Steps: parse → classify → chunk → embed → store vectors → extract entities
    """
    log.info("pipeline.start", document_id=document_id)

    try:
        # ── 1. Load document record ─────────────────────────────
        async def _load_doc():
            async with AsyncSessionLocal() as session:
                from sqlalchemy import select
                result = await session.execute(
                    select(Document).where(Document.id == uuid.UUID(document_id))
                )
                return result.scalar_one_or_none()

        doc = run_async(_load_doc())
        if not doc:
            raise ValueError(f"Document {document_id} not found")

        file_path = storage_service.get_absolute_path(doc.file_path)

        # ── 2. Parse PDF ─────────────────────────────────────────
        run_async(_update_status(document_id, DocumentStatus.PARSING, progress=0.1))
        parsed = parse_pdf(file_path)

        # ── 3. Classify + extract metadata ───────────────────────
        run_async(_update_status(document_id, DocumentStatus.ENRICHING, progress=0.25))
        classification = run_async(classify_document(parsed.full_text[:3000]))

        # ── 4. Chunk ─────────────────────────────────────────────
        run_async(_update_status(document_id, DocumentStatus.CHUNKING, progress=0.40))
        chunks = chunk_document(parsed)

        # ── 5. Embed ─────────────────────────────────────────────
        run_async(_update_status(document_id, DocumentStatus.EMBEDDING, progress=0.55))
        vectors = run_async(embedder_service.embed_chunks(chunks))

        # ── 6. Store in ChromaDB ──────────────────────────────────
        run_async(_update_status(document_id, DocumentStatus.EMBEDDING, progress=0.70))
        collection_name = f"user_{str(doc.user_id).replace('-', '')}_{document_id.replace('-', '')}"
        chroma_ids = run_async(chroma_service.upsert_chunks(
            collection_name=collection_name,
            chunks=chunks,
            vectors=vectors,
            document_id=document_id,
        ))

        # ── 7. Persist chunks to PostgreSQL ──────────────────────
        run_async(_update_status(document_id, DocumentStatus.EMBEDDING, progress=0.82))

        async def _save_chunks():
            async with AsyncSessionLocal() as session:
                db_chunks = [
                    Chunk(
                        document_id=uuid.UUID(document_id),
                        chunk_index=c.chunk_index,
                        text=c.text,
                        page_number=c.page_number,
                        section_heading=c.section_heading,
                        chunk_type=c.chunk_type,
                        chroma_id=chroma_ids[i],
                        token_count=c.token_count,
                        chunk_metadata=c.metadata,
                    )
                    for i, c in enumerate(chunks)
                ]
                session.add_all(db_chunks)
                await session.commit()
                return [str(c.id) for c in db_chunks]

        chunk_ids = run_async(_save_chunks())

        # ── 8. Extract entities (sample first 20 chunks) ─────────
        all_entities: list[dict] = []
        for chunk in chunks[:20]:
            ents = run_async(extract_entities(chunk.text, chunk.page_number))
            all_entities.extend(ents)

        async def _save_entities():
            async with AsyncSessionLocal() as session:
                db_entities = [
                    Entity(
                        document_id=uuid.UUID(document_id),
                        entity_type=e.get("type"),
                        entity_text=e.get("text"),
                        page_number=e.get("page_number"),
                        confidence=e.get("confidence", 1.0),
                    )
                    for e in all_entities
                ]
                session.add_all(db_entities)
                # Update document metadata + type
                await session.execute(
                    update(Document)
                    .where(Document.id == uuid.UUID(document_id))
                    .values(
                        page_count=parsed.total_pages,
                        doc_type=classification.get("doc_type"),
                        doc_metadata={
                            "title": classification.get("title", ""),
                            "author": classification.get("author", ""),
                            "summary": classification.get("summary", ""),
                            "key_dates": classification.get("key_dates", []),
                            "total_chunks": len(chunks),
                        },
                        chroma_collection_id=collection_name,
                    )
                )
                await session.commit()

        run_async(_save_entities())

        # ── 9. Mark ready ─────────────────────────────────────────
        run_async(_update_status(document_id, DocumentStatus.READY, progress=1.0))
        log.info("pipeline.complete", document_id=document_id, chunks=len(chunks))
        return {"document_id": document_id, "chunks": len(chunks), "status": "ready"}

    except Exception as exc:
        log.error("pipeline.failed", document_id=document_id, error=str(exc))
        run_async(_update_status(document_id, DocumentStatus.FAILED, error=str(exc)))
        raise self.retry(exc=exc)
```

---

## Phase 5 — ChromaDB Store

### `app/retrieval/chroma_store.py`
```python
import uuid
from pathlib import Path

import chromadb
import structlog
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.ingestion.chunker import Chunk

log = structlog.get_logger()


class ChromaService:
    def __init__(self):
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def get_or_create_collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        document_id: str,
    ) -> list[str]:
        """Upsert chunk embeddings into ChromaDB. Returns list of chroma IDs."""
        collection = self.get_or_create_collection(collection_name)

        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "chunk_type": c.chunk_type,
                "section_heading": c.section_heading or "",
            }
            for c in chunks
        ]

        # ChromaDB upsert in batches of 500
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[i: i + batch_size],
                embeddings=vectors[i: i + batch_size],
                documents=documents[i: i + batch_size],
                metadatas=metadatas[i: i + batch_size],
            )
            log.info("chroma.upserted", batch=i // batch_size, count=len(ids[i: i + batch_size]))

        return ids

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        n_results: int = 20,
        where: dict | None = None,
    ) -> list[dict]:
        collection = self.get_or_create_collection(collection_name)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(n_results, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        if not results["ids"] or not results["ids"][0]:
            return []
        return [
            {
                "chroma_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 - results["distances"][0][i],  # cosine → similarity
            }
            for i in range(len(results["ids"][0]))
        ]

    def delete_collection(self, collection_name: str) -> None:
        try:
            self.client.delete_collection(collection_name)
            log.info("chroma.collection_deleted", name=collection_name)
        except Exception as e:
            log.warning("chroma.delete_failed", name=collection_name, error=str(e))


chroma_service = ChromaService()
```

---

## Phase 6 — Retrieval & LangGraph RAG Graph

### `app/retrieval/bm25_retriever.py`
```python
from rank_bm25 import BM25Okapi
import structlog

log = structlog.get_logger()


class BM25Retriever:
    """
    In-memory BM25 over a list of chunk dicts.
    Built fresh per query from the document's PostgreSQL chunks.
    For large docs, consider caching this per document.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                chunk = self.chunks[idx].copy()
                chunk["bm25_score"] = float(scores[idx])
                chunk["bm25_rank"] = rank + 1
                results.append(chunk)
        return results
```

### `app/retrieval/hybrid_search.py`
```python
"""
Reciprocal Rank Fusion (RRF) to merge dense vector search + BM25 results.
"""
from collections import defaultdict


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[dict]:
    """
    RRF formula: score = sum(weight / (k + rank)) across retrievers.
    Returns merged + deduplicated list sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, dict] = {}

    for rank, result in enumerate(vector_results, start=1):
        cid = result.get("chroma_id") or result.get("chunk_index", str(rank))
        rrf_scores[cid] += vector_weight / (k + rank)
        chunk_map[cid] = result

    for rank, result in enumerate(bm25_results, start=1):
        cid = result.get("chroma_id") or str(result.get("chunk_index", rank))
        rrf_scores[cid] += bm25_weight / (k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = result

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    merged = []
    for cid in sorted_ids:
        chunk = chunk_map[cid].copy()
        chunk["rrf_score"] = rrf_scores[cid]
        merged.append(chunk)

    return merged
```

### `app/retrieval/query_rewriter.py`
```python
import json

import structlog
from openai import AsyncOpenAI

from app.core.config import settings

log = structlog.get_logger()
client = AsyncOpenAI(api_key=settings.openai_api_key)

MULTI_QUERY_PROMPT = """You are an expert at reformulating search queries.
Given a user question, generate 3 different versions that capture different angles.
Respond ONLY with JSON: {"queries": ["...", "...", "..."]}"""

HYDE_PROMPT = """Given the question below, write a short (3-4 sentence) hypothetical 
document excerpt that would perfectly answer it. This will be used for semantic search.
Do NOT add any preamble."""


async def expand_query(query: str) -> list[str]:
    """Returns original query + 3 rephrased versions (multi-query expansion)."""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": MULTI_QUERY_PROMPT},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        data = json.loads(response.choices[0].message.content)
        variants = data.get("queries", [])
        return [query] + variants[:3]
    except Exception as e:
        log.warning("query_rewriter.expand_failed", error=str(e))
        return [query]


async def generate_hyde_embedding_text(query: str) -> str:
    """Generate hypothetical document for HyDE retrieval."""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": HYDE_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        log.warning("query_rewriter.hyde_failed", error=str(e))
        return query
```

### `app/retrieval/reranker.py`
```python
import structlog
from sentence_transformers import CrossEncoder

from app.core.config import settings

log = structlog.get_logger()

_model: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        log.info("reranker.loading", model=settings.reranker_model)
        _model = CrossEncoder(settings.reranker_model)
    return _model


def rerank(query: str, chunks: list[dict], top_n: int = None) -> list[dict]:
    """Rerank chunks using cross-encoder. Returns top_n chunks sorted by score."""
    if not settings.enable_reranker or not chunks:
        return chunks[:top_n or settings.reranker_top_n]

    top_n = top_n or settings.reranker_top_n
    model = get_reranker()
    pairs = [(query, c["text"]) for c in chunks]

    try:
        scores = model.predict(pairs).tolist()
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = score

        ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        log.info("reranker.complete", input=len(chunks), output=top_n,
                 top_score=round(ranked[0]["rerank_score"], 3) if ranked else 0)
        return ranked[:top_n]

    except Exception as e:
        log.error("reranker.failed", error=str(e))
        return chunks[:top_n]
```

### `app/retrieval/graph.py`
```python
"""
LangGraph RAG pipeline:
  query_rewriter → hybrid_retriever → reranker → answer_generator

State flows through typed TypedDict.
W&B Weave decorators capture every node for tracing.
"""
from __future__ import annotations

import time
from typing import Annotated, Any, AsyncIterator, TypedDict

import weave
import structlog
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.core.config import settings
from app.ingestion.embedder import embedder_service
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.chroma_store import chroma_service
from app.retrieval.hybrid_search import reciprocal_rank_fusion
from app.retrieval.query_rewriter import expand_query, generate_hyde_embedding_text
from app.retrieval.reranker import rerank

log = structlog.get_logger()


class RAGState(TypedDict):
    query: str
    document_ids: list[str]
    collection_names: list[str]
    db_chunks: list[dict]           # chunks from PostgreSQL for BM25
    expanded_queries: list[str]
    hyde_text: str
    vector_results: list[dict]
    bm25_results: list[dict]
    fused_results: list[dict]
    reranked_results: list[dict]
    answer: str
    citations: list[dict]
    confidence: float
    low_confidence: bool


ANSWER_SYSTEM_PROMPT = """You are an expert document analyst for an IDP system.
Answer the user's question using ONLY the provided document excerpts.

Rules:
1. Base your answer strictly on the provided context.
2. For every claim, cite the source using [Page X] format.
3. If the context doesn't contain enough information, say so explicitly.
4. Be precise and concise.
5. When answering about numbers, dates, or names — quote them exactly.

Context:
{context}"""


@weave.op()
async def node_rewrite_query(state: RAGState) -> dict:
    log.info("rag.rewrite_query", query=state["query"])
    expanded = await expand_query(state["query"])
    hyde_text = await generate_hyde_embedding_text(state["query"])
    return {"expanded_queries": expanded, "hyde_text": hyde_text}


@weave.op()
async def node_hybrid_retrieve(state: RAGState) -> dict:
    """Run vector search (with HyDE + all expanded queries) and BM25 in parallel."""
    import asyncio

    # Vector search: embed HyDE text + each expanded query, union results
    async def vector_search_one(text: str) -> list[dict]:
        vec = await embedder_service.embed_query(text)
        all_results = []
        for cname in state["collection_names"]:
            results = chroma_service.search(cname, vec, n_results=settings.retrieval_top_k)
            all_results.extend(results)
        return all_results

    search_texts = [state["hyde_text"]] + state["expanded_queries"]
    all_vector_results_nested = await asyncio.gather(*[vector_search_one(t) for t in search_texts])

    # Deduplicate by chroma_id, keep highest score
    seen: dict[str, dict] = {}
    for results in all_vector_results_nested:
        for r in results:
            cid = r["chroma_id"]
            if cid not in seen or r["score"] > seen[cid]["score"]:
                seen[cid] = r
    vector_results = list(seen.values())

    # BM25 search over PostgreSQL chunks
    bm25_retriever = BM25Retriever(state["db_chunks"])
    bm25_results = bm25_retriever.search(state["query"], top_k=settings.retrieval_top_k)

    log.info("rag.retrieved", vector=len(vector_results), bm25=len(bm25_results))
    return {"vector_results": vector_results, "bm25_results": bm25_results}


@weave.op()
async def node_fuse_and_rerank(state: RAGState) -> dict:
    fused = reciprocal_rank_fusion(
        state["vector_results"],
        state["bm25_results"],
    )
    reranked = rerank(state["query"], fused, top_n=settings.reranker_top_n)

    # Confidence = average of top reranked scores (or RRF scores if no reranker)
    scores = [r.get("rerank_score") or r.get("rrf_score", 0) for r in reranked]
    confidence = sum(scores) / len(scores) if scores else 0.0

    log.info("rag.fused", fused=len(fused), reranked=len(reranked),
             confidence=round(confidence, 3))
    return {
        "fused_results": fused,
        "reranked_results": reranked,
        "confidence": confidence,
        "low_confidence": confidence < settings.confidence_threshold,
    }


@weave.op()
async def node_generate_answer(state: RAGState) -> dict:
    if state["low_confidence"]:
        return {
            "answer": "I don't have enough relevant information in the provided documents to answer this question accurately.",
            "citations": [],
        }

    context_parts = []
    for chunk in state["reranked_results"]:
        page = chunk.get("metadata", {}).get("page_number", "?")
        context_parts.append(f"[Page {page}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        temperature=0,
        streaming=False,
        openai_api_key=settings.openai_api_key,
    )

    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": state["query"]},
    ]

    response = await llm.ainvoke(messages)
    answer = response.content

    citations = [
        {
            "chroma_id": c.get("chroma_id"),
            "page": c.get("metadata", {}).get("page_number"),
            "snippet": c["text"][:200],
            "score": round(c.get("rerank_score") or c.get("rrf_score", 0), 3),
        }
        for c in state["reranked_results"]
    ]

    return {"answer": answer, "citations": citations}


def build_rag_graph() -> StateGraph:
    graph = StateGraph(RAGState)
    graph.add_node("rewrite_query", node_rewrite_query)
    graph.add_node("hybrid_retrieve", node_hybrid_retrieve)
    graph.add_node("fuse_and_rerank", node_fuse_and_rerank)
    graph.add_node("generate_answer", node_generate_answer)

    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "fuse_and_rerank")
    graph.add_edge("fuse_and_rerank", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


rag_graph = build_rag_graph()
```

---

## Phase 7 — Pydantic Schemas

### `app/schemas/documents.py`
```python
from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    QUEUED = "queued"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    ENRICHING = "enriching"
    READY = "ready"
    FAILED = "failed"


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    filename: str
    status: DocumentStatus
    message: str


class DocumentStatusResponse(BaseModel):
    document_id: UUID
    status: DocumentStatus
    progress: float = Field(ge=0.0, le=1.0)
    page_count: int | None
    doc_type: str | None
    error_message: str | None
    created_at: datetime
    processed_at: datetime | None


class DocumentListItem(BaseModel):
    document_id: UUID
    filename: str
    status: DocumentStatus
    doc_type: str | None
    page_count: int | None
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentDetail(DocumentListItem):
    doc_metadata: dict
    progress: float
    processed_at: datetime | None
```

### `app/schemas/query.py`
```python
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[UUID] = Field(min_length=1, max_length=10)
    session_id: UUID | None = None
    stream: bool = False


class Citation(BaseModel):
    chroma_id: str | None
    page: int | None
    snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    session_id: UUID
    turn_id: UUID
    latency_ms: int
    weave_trace_id: str | None


class QueryHistoryItem(BaseModel):
    turn_id: UUID
    question: str
    answer: str
    citations: list[Citation]
    confidence: float
    created_at: datetime
```

### `app/schemas/auth.py`
```python
from pydantic import BaseModel, EmailStr


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
```

---

## Phase 8 — FastAPI Routes

### `app/api/deps.py`
```python
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_token
from app.db.base import get_db
from app.db.models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        user_id = decode_token(token)
    except JWTError:
        raise credentials_exception

    result = await db.execute(select(User).where(User.id == UUID(user_id)))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise credentials_exception
    return user
```

### `app/api/routes/auth.py`
```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, hash_password, verify_password
from app.db.base import get_db
from app.db.models import User
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(email=body.email, hashed_password=hash_password(body.password))
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return TokenResponse(access_token=create_access_token(str(user.id)))


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(access_token=create_access_token(str(user.id)))
```

### `app/api/routes/documents.py`
```python
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db.base import get_db
from app.db.models import Document, DocumentStatus, User
from app.ingestion.pipeline import task_ingest_document
from app.ingestion.storage import storage_service
from app.retrieval.chroma_store import chroma_service
from app.schemas.documents import (
    DocumentDetail, DocumentListItem,
    DocumentStatusResponse, DocumentUploadResponse
)
import structlog

log = structlog.get_logger()
router = APIRouter(prefix="/documents", tags=["documents"])

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


@router.post("/upload", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 100MB limit")

    # Save to local storage
    stored_path = await storage_service.save(
        contents, str(current_user.id), file.filename
    )

    # Create DB record
    doc = Document(
        user_id=current_user.id,
        filename=f"{uuid.uuid4()}_{file.filename}",
        original_filename=file.filename,
        file_path=stored_path,
        file_size_bytes=len(contents),
        status=DocumentStatus.QUEUED,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # Enqueue Celery task
    task = task_ingest_document.delay(str(doc.id))
    doc.celery_task_id = task.id
    await db.commit()

    log.info("document.queued", document_id=str(doc.id), filename=file.filename,
             user_id=str(current_user.id))

    return DocumentUploadResponse(
        document_id=doc.id,
        filename=file.filename,
        status=DocumentStatus.QUEUED,
        message="Document queued for processing",
    )


@router.get("", response_model=list[DocumentListItem])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 20,
):
    result = await db.execute(
        select(Document)
        .where(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    docs = result.scalars().all()
    return [DocumentListItem(
        document_id=d.id,
        filename=d.original_filename,
        status=d.status,
        doc_type=d.doc_type,
        page_count=d.page_count,
        created_at=d.created_at,
    ) for d in docs]


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == current_user.id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentStatusResponse(
        document_id=doc.id,
        status=doc.status,
        progress=doc.progress,
        page_count=doc.page_count,
        doc_type=doc.doc_type,
        error_message=doc.error_message,
        created_at=doc.created_at,
        processed_at=doc.processed_at,
    )


@router.get("/{document_id}/events")
async def document_events(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """SSE stream for real-time ingestion progress."""
    import asyncio

    async def event_generator() -> AsyncIterator[str]:
        while True:
            result = await db.execute(
                select(Document).where(
                    Document.id == document_id,
                    Document.user_id == current_user.id,
                )
            )
            doc = result.scalar_one_or_none()
            if not doc:
                yield "data: {\"error\": \"Document not found\"}\n\n"
                break

            payload = (
                f'data: {{"status": "{doc.status}", "progress": {doc.progress}}}\n\n'
            )
            yield payload

            if doc.status in (DocumentStatus.READY, DocumentStatus.FAILED):
                break
            await asyncio.sleep(1.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == current_user.id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.chroma_collection_id:
        chroma_service.delete_collection(doc.chroma_collection_id)

    await storage_service.delete(doc.file_path)
    await db.delete(doc)
    await db.commit()
```

### `app/api/routes/query.py`
```python
import time
import uuid
from typing import AsyncIterator

import structlog
import weave
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.config import settings
from app.db.base import get_db
from app.db.models import Chunk, Document, DocumentStatus, QuerySession, QueryTurn, User
from app.retrieval.graph import RAGState, rag_graph
from app.schemas.query import Citation, QueryRequest, QueryResponse

log = structlog.get_logger()
router = APIRouter(prefix="/query", tags=["query"])


async def _load_rag_state(
    document_ids: list[uuid.UUID],
    user_id: uuid.UUID,
    db: AsyncSession,
) -> tuple[list[str], list[dict]]:
    """
    Verify all docs belong to user and are READY.
    Returns (collection_names, db_chunks_for_bm25).
    """
    result = await db.execute(
        select(Document).where(
            Document.id.in_(document_ids),
            Document.user_id == user_id,
        )
    )
    docs = result.scalars().all()

    if len(docs) != len(document_ids):
        raise HTTPException(status_code=404, detail="One or more documents not found")

    not_ready = [d for d in docs if d.status != DocumentStatus.READY]
    if not_ready:
        raise HTTPException(
            status_code=409,
            detail=f"Documents not ready: {[str(d.id) for d in not_ready]}"
        )

    collection_names = [d.chroma_collection_id for d in docs if d.chroma_collection_id]

    # Load chunks for BM25
    chunks_result = await db.execute(
        select(Chunk).where(Chunk.document_id.in_(document_ids))
    )
    db_chunks = [
        {
            "chroma_id": c.chroma_id,
            "text": c.text,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "chunk_type": c.chunk_type,
            "document_id": str(c.document_id),
        }
        for c in chunks_result.scalars().all()
    ]

    return collection_names, db_chunks


@router.post("", response_model=QueryResponse)
async def query_documents(
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    start_ms = int(time.time() * 1000)

    collection_names, db_chunks = await _load_rag_state(
        body.document_ids, current_user.id, db
    )

    # Get or create session
    session_id = body.session_id
    if not session_id:
        session = QuerySession(
            user_id=current_user.id,
            document_ids=[str(d) for d in body.document_ids],
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)
        session_id = session.id
    
    # Run LangGraph
    with weave.attributes({"session_id": str(session_id), "user_id": str(current_user.id)}):
        initial_state: RAGState = {
            "query": body.question,
            "document_ids": [str(d) for d in body.document_ids],
            "collection_names": collection_names,
            "db_chunks": db_chunks,
            "expanded_queries": [],
            "hyde_text": "",
            "vector_results": [],
            "bm25_results": [],
            "fused_results": [],
            "reranked_results": [],
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "low_confidence": False,
        }
        final_state = await rag_graph.ainvoke(initial_state)

    latency_ms = int(time.time() * 1000) - start_ms

    # Persist turn
    turn = QueryTurn(
        session_id=session_id,
        question=body.question,
        answer=final_state["answer"],
        citations=final_state["citations"],
        confidence=final_state["confidence"],
        retrieved_chunks=len(final_state["reranked_results"]),
        latency_ms=latency_ms,
    )
    db.add(turn)
    await db.commit()
    await db.refresh(turn)

    return QueryResponse(
        answer=final_state["answer"],
        citations=[Citation(**c) for c in final_state["citations"]],
        confidence=final_state["confidence"],
        session_id=session_id,
        turn_id=turn.id,
        latency_ms=latency_ms,
        weave_trace_id=None,
    )


@router.post("/stream")
async def query_stream(
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Streaming Q&A via SSE.
    Runs retrieval graph first, then streams LLM answer token by token.
    """
    collection_names, db_chunks = await _load_rag_state(
        body.document_ids, current_user.id, db
    )

    async def event_generator() -> AsyncIterator[str]:
        import json
        from app.retrieval.graph import (
            node_fuse_and_rerank, node_hybrid_retrieve, node_rewrite_query
        )
        from app.retrieval.graph import ANSWER_SYSTEM_PROMPT

        state: RAGState = {
            "query": body.question,
            "document_ids": [str(d) for d in body.document_ids],
            "collection_names": collection_names,
            "db_chunks": db_chunks,
            "expanded_queries": [],
            "hyde_text": "",
            "vector_results": [],
            "bm25_results": [],
            "fused_results": [],
            "reranked_results": [],
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "low_confidence": False,
        }

        # Step through retrieval nodes
        yield f"data: {json.dumps({'type': 'status', 'status': 'rewriting_query'})}\n\n"
        state.update(await node_rewrite_query(state))

        yield f"data: {json.dumps({'type': 'status', 'status': 'retrieving'})}\n\n"
        state.update(await node_hybrid_retrieve(state))

        yield f"data: {json.dumps({'type': 'status', 'status': 'reranking'})}\n\n"
        state.update(await node_fuse_and_rerank(state))

        # Stream the answer
        yield f"data: {json.dumps({'type': 'status', 'status': 'generating'})}\n\n"

        if state["low_confidence"]:
            msg = "I don't have enough relevant information in the provided documents."
            yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"
        else:
            context_parts = [
                f"[Page {c.get('metadata', {}).get('page_number', '?')}]\n{c['text']}"
                for c in state["reranked_results"]
            ]
            context = "\n\n---\n\n".join(context_parts)
            llm = ChatOpenAI(
                model=settings.openai_chat_model,
                temperature=0,
                streaming=True,
                openai_api_key=settings.openai_api_key,
            )
            messages = [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": body.question},
            ]
            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

        citations = [
            {
                "page": c.get("metadata", {}).get("page_number"),
                "snippet": c["text"][:150],
                "score": round(c.get("rerank_score") or c.get("rrf_score", 0), 3),
            }
            for c in state["reranked_results"]
        ]
        yield f"data: {json.dumps({'type': 'citations', 'citations': citations, 'confidence': state['confidence']})}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/history/{session_id}")
async def get_query_history(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(QuerySession).where(
            QuerySession.id == session_id,
            QuerySession.user_id == current_user.id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    turns_result = await db.execute(
        select(QueryTurn)
        .where(QueryTurn.session_id == session_id)
        .order_by(QueryTurn.created_at)
    )
    return turns_result.scalars().all()
```

### `app/api/routes/health.py`
```python
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def readiness():
    # TODO: add DB + Redis ping when Docker is set up
    return {"status": "ready"}
```

---

## Phase 9 — FastAPI App Factory

### `app/main.py`
```python
import weave
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException

from app.core.config import settings
from app.core.exceptions import http_exception_handler
from app.core.logging import setup_logging
from app.api.routes import auth, documents, query, health

log = structlog.get_logger()


def create_app() -> FastAPI:
    setup_logging()

    # Initialize W&B Weave tracing
    if settings.wandb_api_key:
        weave.init(settings.wandb_project)
        log.info("weave.initialized", project=settings.wandb_project)

    app = FastAPI(
        title="IDP RAG System",
        description="Intelligent Document Processing with RAG pipeline",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    app.add_exception_handler(HTTPException, http_exception_handler)

    # Prometheus metrics (off by default, enable on server)
    if settings.enable_metrics:
        from prometheus_fastapi_instrumentator import Instrumentator
        Instrumentator().instrument(app).expose(app)

    # Routers
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(documents.router)
    app.include_router(query.router)

    @app.on_event("startup")
    async def startup():
        log.info("app.started", environment=settings.environment)

    @app.on_event("shutdown")
    async def shutdown():
        log.info("app.shutdown")

    return app


app = create_app()
```

---

## Phase 10 — Celery Worker

### `worker/celery_app.py`
```python
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "idp_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.ingestion.pipeline"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "app.ingestion.pipeline.task_ingest_document": {"queue": "ingestion"},
    },
)
```

---

## Phase 11 — MCP Server

The MCP server exposes your RAG pipeline as tools that any MCP-compatible client (Cursor, Claude Desktop) can call directly. This means while you're in Cursor, you can ask the AI to search your document store or check processing status without leaving the IDE.

### `mcp_server/server.py`
```python
"""
FastMCP server exposing IDP RAG capabilities as MCP tools.

Run alongside the FastAPI app:
    python -m mcp_server.server

Connect in Cursor: Settings → MCP → add server at http://localhost:8001

Available tools:
  - search_documents: semantic + keyword search across uploaded docs
  - ask_document: full RAG Q&A on a specific document
  - list_documents: list all processed documents for a user
  - get_document_status: check processing progress
  - get_document_entities: retrieve extracted entities (people, orgs, dates)
"""
import asyncio
import httpx
from typing import Any

from fastmcp import FastMCP

from app.core.config import settings

mcp = FastMCP(
    name="IDP RAG Server",
    instructions="""
    You have access to an Intelligent Document Processing (IDP) system.
    Users can upload large PDFs (contracts, invoices, reports, research papers).
    Documents are processed via OCR, chunking, and embedding.
    You can search across documents, ask questions, and extract structured data.
    Always cite page numbers when answering questions about document content.
    """,
)

API_BASE = "http://localhost:8000"

# In a real multi-user deployment, each tool call would pass a JWT.
# For local dev, hardcode a dev token or read from env.
DEV_TOKEN = ""  # Set this after calling /auth/login during development


def _headers() -> dict:
    if DEV_TOKEN:
        return {"Authorization": f"Bearer {DEV_TOKEN}"}
    return {}


@mcp.tool()
async def search_documents(
    query: str,
    document_id: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Perform hybrid semantic + keyword search over a processed document.

    Args:
        query: The search query
        document_id: UUID of the document to search
        top_k: Number of results to return (default 5, max 20)

    Returns:
        Dict with 'results' list, each containing text, page_number, score
    """
    from app.ingestion.embedder import embedder_service
    from app.retrieval.chroma_store import chroma_service
    from app.retrieval.bm25_retriever import BM25Retriever
    from app.retrieval.hybrid_search import reciprocal_rank_fusion
    from app.retrieval.reranker import rerank
    import uuid
    from sqlalchemy import select
    from app.db.base import AsyncSessionLocal
    from app.db.models import Document, Chunk

    async with AsyncSessionLocal() as db:
        doc_result = await db.execute(
            select(Document).where(Document.id == uuid.UUID(document_id))
        )
        doc = doc_result.scalar_one_or_none()
        if not doc:
            return {"error": f"Document {document_id} not found"}

        if not doc.chroma_collection_id:
            return {"error": "Document not yet processed"}

        # Load chunks for BM25
        chunks_result = await db.execute(
            select(Chunk).where(Chunk.document_id == uuid.UUID(document_id))
        )
        db_chunks = [
            {"chroma_id": c.chroma_id, "text": c.text, "page_number": c.page_number}
            for c in chunks_result.scalars().all()
        ]

    # Vector search
    query_vec = await embedder_service.embed_query(query)
    vector_results = chroma_service.search(
        doc.chroma_collection_id, query_vec, n_results=min(top_k * 2, 20)
    )

    # BM25
    bm25 = BM25Retriever(db_chunks)
    bm25_results = bm25.search(query, top_k=min(top_k * 2, 20))

    # Fuse + rerank
    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    reranked = rerank(query, fused, top_n=top_k)

    return {
        "query": query,
        "document_id": document_id,
        "results": [
            {
                "text": r["text"],
                "page_number": r.get("metadata", {}).get("page_number"),
                "score": round(r.get("rerank_score") or r.get("rrf_score", 0), 3),
            }
            for r in reranked
        ],
    }


@mcp.tool()
async def ask_document(
    question: str,
    document_id: str,
) -> dict[str, Any]:
    """
    Ask a question and get a RAG-powered answer from a specific document.

    Args:
        question: The question to ask
        document_id: UUID of the document to query

    Returns:
        Dict with 'answer', 'citations' (page numbers), 'confidence'
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE}/query",
            json={
                "question": question,
                "document_ids": [document_id],
            },
            headers=_headers(),
            timeout=60.0,
        )
        if response.status_code != 200:
            return {"error": response.text}
        return response.json()


@mcp.tool()
async def list_documents() -> dict[str, Any]:
    """
    List all processed documents available in the system.

    Returns:
        Dict with 'documents' list, each containing id, filename, status, doc_type, page_count
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE}/documents",
            headers=_headers(),
            timeout=15.0,
        )
        if response.status_code != 200:
            return {"error": response.text}
        return {"documents": response.json()}


@mcp.tool()
async def get_document_status(document_id: str) -> dict[str, Any]:
    """
    Check the processing status and progress of a document.

    Args:
        document_id: UUID of the document

    Returns:
        Dict with status, progress (0.0-1.0), page_count, doc_type, error_message
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE}/documents/{document_id}/status",
            headers=_headers(),
            timeout=15.0,
        )
        if response.status_code != 200:
            return {"error": response.text}
        return response.json()


@mcp.tool()
async def get_document_entities(document_id: str) -> dict[str, Any]:
    """
    Get named entities extracted from a document.
    Includes people, organizations, dates, monetary amounts, locations.

    Args:
        document_id: UUID of the document

    Returns:
        Dict with 'entities' grouped by type
    """
    import uuid
    from collections import defaultdict
    from sqlalchemy import select
    from app.db.base import AsyncSessionLocal
    from app.db.models import Entity

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Entity).where(Entity.document_id == uuid.UUID(document_id))
        )
        entities = result.scalars().all()

    grouped: dict[str, list] = defaultdict(list)
    for e in entities:
        grouped[e.entity_type].append({
            "text": e.entity_text,
            "page": e.page_number,
            "confidence": round(e.confidence or 0, 3),
        })

    return {
        "document_id": document_id,
        "total_entities": len(entities),
        "entities": dict(grouped),
    }


@mcp.tool()
async def compare_documents(
    question: str,
    document_id_a: str,
    document_id_b: str,
) -> dict[str, Any]:
    """
    Compare two documents by asking the same question against both and
    returning a structured comparison.

    Args:
        question: What to compare across the two documents
        document_id_a: First document UUID
        document_id_b: Second document UUID

    Returns:
        Dict with answers from both documents and a synthesized comparison
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE}/query",
            json={
                "question": question,
                "document_ids": [document_id_a, document_id_b],
            },
            headers=_headers(),
            timeout=60.0,
        )
        if response.status_code != 200:
            return {"error": response.text}
        return response.json()


if __name__ == "__main__":
    import uvicorn
    # FastMCP exposes itself as an HTTP server
    mcp.run(
        transport="sse",
        host=settings.mcp_server_host,
        port=settings.mcp_server_port,
    )
```

---

## Phase 12 — Alembic Setup

### `alembic.ini` (key settings)
```ini
[alembic]
script_location = alembic
sqlalchemy.url = %(DATABASE_SYNC_URL)s
```

### `alembic/env.py`
```python
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

from app.core.config import settings
from app.db.base import Base
from app.db import models  # noqa: F401 — import all models

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_sync_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

---

## How to Run Everything (No Docker)

### Prerequisites
```bash
# Install PostgreSQL and Redis locally (macOS)
brew install postgresql redis
brew services start postgresql
brew services start redis

# Install spaCy model for NER
python -m spacy download en_core_web_sm

# Install Python deps
pip install -e ".[dev]"
```

### Setup
```bash
# 1. Copy env
cp .env.example .env
# Fill in OPENAI_API_KEY, WANDB_API_KEY, DATABASE_URL, SECRET_KEY

# 2. Create database
createdb idprag

# 3. Run migrations
alembic upgrade head

# 4. Create required directories
mkdir -p uploads logs chroma_store
```

### Run (3 terminals)
```bash
# Terminal 1 — FastAPI
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Celery worker
celery -A worker.celery_app worker -Q ingestion --loglevel=info

# Terminal 3 — MCP server (optional, for Cursor integration)
python -m mcp_server.server
```

### Connect MCP to Cursor
```
Cursor Settings → Features → MCP Servers → Add Server
Name: IDP RAG
URL: http://localhost:8001/sse
```

---

## API Usage Walkthrough

```bash
# 1. Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"dev@test.com","password":"password123"}'
# → {"access_token": "eyJ...", "token_type": "bearer"}

# 2. Upload PDF
curl -X POST http://localhost:8000/documents/upload \
  -H "Authorization: Bearer eyJ..." \
  -F "file=@/path/to/contract.pdf"
# → {"document_id": "uuid", "status": "queued"}

# 3. Poll status (or watch SSE stream)
curl http://localhost:8000/documents/{id}/status \
  -H "Authorization: Bearer eyJ..."
# → {"status": "ready", "progress": 1.0, "page_count": 247}

# 4. Ask a question
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the payment terms?","document_ids":["uuid"]}'

# 5. Streaming answer
curl -N -X POST http://localhost:8000/query/stream \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize section 3","document_ids":["uuid"]}'
```

---

## Implementation Order for Cursor

Work through these files in this exact order to avoid import errors:

1. `pyproject.toml` + `.env` — dependencies and env vars
2. `app/core/config.py` — settings first, everything imports this
3. `app/core/logging.py` → `app/core/security.py` → `app/core/exceptions.py`
4. `app/db/base.py` → `app/db/models.py`
5. Run `alembic init alembic` then write `alembic/env.py` → `alembic upgrade head`
6. `app/ingestion/storage.py`
7. `app/ingestion/pdf_parser.py` → `chunker.py` → `embedder.py` → `enricher.py`
8. `app/retrieval/chroma_store.py`
9. `app/ingestion/pipeline.py` (imports storage + parser + chunker + embedder + enricher + chroma)
10. `worker/celery_app.py`
11. `app/retrieval/bm25_retriever.py` → `hybrid_search.py` → `query_rewriter.py` → `reranker.py`
12. `app/retrieval/graph.py` (LangGraph — imports all retrieval modules)
13. `app/schemas/` — all three schema files
14. `app/api/deps.py`
15. `app/api/routes/` — health → auth → documents → query
16. `app/main.py`
17. `mcp_server/server.py`
18. `tests/`

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| LangGraph over bare LangChain | Explicit state machine — easy to add/remove nodes, debug each step |
| Per-document ChromaDB collection | User A's vectors never mix with User B's; simple scoped deletion |
| BM25 from PostgreSQL chunks | Avoids second vector store; chunks already in DB for citation linking |
| Celery task as single function (not chain) | Simpler progress tracking; partial failures handled with retry |
| HyDE + multi-query | Dramatically improves recall on long complex documents |
| Cross-encoder reranker locally | Free, no API cost, runs in ~100ms on CPU for top-20 chunks |
| MCP alongside REST | REST for application use; MCP for Cursor/Claude Desktop developer tooling |
| W&B Weave traces | Every LLM call logged with inputs/outputs/latency — debug retrieval quality from day one |
