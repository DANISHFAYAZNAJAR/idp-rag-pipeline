import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
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

    documents = relationship(
        "Document", back_populates="user", cascade="all, delete-orphan"
    )
    sessions = relationship(
        "QuerySession", back_populates="user", cascade="all, delete-orphan"
    )


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size_bytes = Column(Integer)
    page_count = Column(Integer)
    status = Column(
        Enum(DocumentStatus, native_enum=False, length=50),
        default=DocumentStatus.QUEUED,
        index=True,
    )
    progress = Column(Float, default=0.0)
    doc_type = Column(
        Enum(DocumentType, native_enum=False, length=50),
        default=DocumentType.UNKNOWN,
    )
    doc_metadata = Column(JSON, default=dict)
    error_message = Column(Text)
    celery_task_id = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=utcnow)
    processed_at = Column(DateTime(timezone=True))

    user = relationship("User", back_populates="documents")
    chunks = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )
    entities = relationship(
        "Entity", back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True
    )
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    page_number = Column(Integer)
    section_heading = Column(String(500))
    chunk_type = Column(String(50), default="text")
    chroma_id = Column(String(255), index=True)
    token_count = Column(Integer)
    chunk_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    document = relationship("Document", back_populates="chunks")


class Entity(Base):
    __tablename__ = "entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True
    )
    entity_type = Column(String(100))
    entity_text = Column(String(1000))
    page_number = Column(Integer)
    confidence = Column(Float)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    document = relationship("Document", back_populates="entities")


class QuerySession(Base):
    __tablename__ = "query_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    document_ids = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    user = relationship("User", back_populates="sessions")
    turns = relationship(
        "QueryTurn", back_populates="session", cascade="all, delete-orphan"
    )


class QueryTurn(Base):
    __tablename__ = "query_turns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True), ForeignKey("query_sessions.id"), nullable=False, index=True
    )
    question = Column(Text, nullable=False)
    answer = Column(Text)
    citations = Column(JSON, default=list)
    confidence = Column(Float)
    retrieved_chunks = Column(Integer)
    latency_ms = Column(Integer)
    weave_trace_id = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=utcnow)

    session = relationship("QuerySession", back_populates="turns")
