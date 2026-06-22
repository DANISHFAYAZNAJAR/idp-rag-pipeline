from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[UUID] = Field(min_length=1, max_length=50)
    session_id: UUID | None = None
    stream: bool = False
    chat_history: list[dict] = Field(default_factory=list)


class Citation(BaseModel):
    chroma_id: str | None = None
    page: int | None = None
    snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    session_id: UUID
    turn_id: UUID
    latency_ms: int
    weave_trace_id: str | None = None


class QueryHistoryItem(BaseModel):
    turn_id: UUID
    question: str
    answer: str
    citations: list[Citation]
    confidence: float
    created_at: datetime

