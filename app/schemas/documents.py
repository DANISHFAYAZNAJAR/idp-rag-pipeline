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
    entities_status: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentDetail(DocumentListItem):
    doc_metadata: dict
    progress: float
    processed_at: datetime | None

