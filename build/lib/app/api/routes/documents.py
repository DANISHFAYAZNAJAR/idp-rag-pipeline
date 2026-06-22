import uuid
from typing import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.config import settings
from app.db.base import get_db
from app.db.models import Document, DocumentStatus, User
from app.ingestion.document_parser import SUPPORTED_EXTENSIONS, is_supported_filename
from app.ingestion.pipeline import task_ingest_document
from app.ingestion.storage import storage_service
from app.retrieval.chroma_store import chroma_service
from app.schemas.documents import (
    DocumentListItem,
    DocumentStatusResponse,
    DocumentUploadResponse,
)

log = structlog.get_logger()
router = APIRouter(prefix="/documents", tags=["documents"])

limiter = Limiter(key_func=get_remote_address)

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


async def _upload_document_impl(
    request: Request,
    file: UploadFile,
    db: AsyncSession,
    current_user: User,
) -> DocumentUploadResponse:
    if not file.filename or not is_supported_filename(file.filename):
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported formats: {supported}",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 100MB limit")

    stored_path = await storage_service.save(contents, str(current_user.id), file.filename)

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

    task = task_ingest_document.delay(str(doc.id))
    doc.celery_task_id = task.id
    await db.commit()

    log.info(
        "document.queued",
        document_id=str(doc.id),
        filename=file.filename,
        user_id=str(current_user.id),
    )

    return DocumentUploadResponse(
        document_id=doc.id,
        filename=file.filename,
        status=DocumentStatus.QUEUED,
        message="Document queued for processing",
    )


async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await _upload_document_impl(request, file, db, current_user)


if settings.environment != "development":
    upload_document = limiter.limit(lambda: f"{settings.rate_limit_uploads_per_hour}/hour")(
        upload_document
    )

router.add_api_route(
    "/upload",
    upload_document,
    methods=["POST"],
    response_model=DocumentUploadResponse,
    status_code=202,
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
    return [
        DocumentListItem(
            document_id=d.id,
            filename=d.original_filename,
            status=d.status,
            doc_type=d.doc_type,
            page_count=d.page_count,
            entities_status=(d.doc_metadata or {}).get("entities_status"),
            created_at=d.created_at,
        )
        for d in docs
    ]


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.user_id == current_user.id)
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
    current_user: User = Depends(get_current_user),
):
    import asyncio

    from app.db.base import AsyncSessionLocal

    async def event_generator() -> AsyncIterator[str]:
        while True:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(Document).where(
                        Document.id == document_id,
                        Document.user_id == current_user.id,
                    )
                )
                doc = result.scalar_one_or_none()

            if not doc:
                yield 'data: {"error": "Document not found"}\n\n'
                break

            yield f'data: {{"status": "{doc.status}", "progress": {doc.progress}}}\n\n'
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
        select(Document).where(Document.id == document_id, Document.user_id == current_user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    collection_name = settings.user_collection_name(str(current_user.id))
    chroma_service.delete_by_document(collection_name=collection_name, document_id=str(doc.id))
    await storage_service.delete(doc.file_path)

    await db.delete(doc)
    await db.commit()

