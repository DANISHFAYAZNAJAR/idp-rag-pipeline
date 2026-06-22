"""
Celery tasks for document ingestion.

Core ingest (parse → chunk → embed) marks the document ready for chat.
Entity extraction (NER) runs in a background thread so ingest jobs are not blocked.
"""
import asyncio
import threading
import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import delete, select, update

from app.core.config import settings
from app.db.base import AsyncSessionLocal, engine
from app.db.sync_base import SyncSessionLocal
from app.db.models import Chunk, Document, DocumentStatus, Entity
from app.ingestion.chunker import ParsedChunk, chunk_document
from app.ingestion.embedder import embedder_service
from app.ingestion.enricher import classify_document, extract_entities_all_chunks
from app.ingestion.document_parser import parse_document
from app.ingestion.storage import storage_service
from app.retrieval.chroma_store import chroma_service
from worker.celery_app import celery_app

log = structlog.get_logger()

# Solo Celery worker runs one task at a time — keep NER off the task queue.
_enrich_thread_slot = threading.Semaphore(1)


async def _update_status(
    doc_id: str,
    status: DocumentStatus,
    progress: float | None = None,
    error: str | None = None,
) -> None:
    doc_uuid = _parse_document_id(doc_id)
    async with AsyncSessionLocal() as session:
        values: dict = {"status": status.value}
        if progress is not None:
            values["progress"] = progress
        if error:
            values["error_message"] = error
        if status == DocumentStatus.READY:
            values["processed_at"] = datetime.now(timezone.utc)
        await session.execute(
            update(Document).where(Document.id == doc_uuid).values(**values)
        )
        await session.commit()


def _parse_document_id(document_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(document_id).strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid document_id '{document_id}'. "
            "Use the UUID from POST /documents/upload or GET /documents."
        ) from exc


def _chunks_from_db(rows: list[Chunk]) -> list[ParsedChunk]:
    return [
        ParsedChunk(
            text=row.text,
            page_number=row.page_number or 0,
            chunk_index=row.chunk_index,
            section_heading=row.section_heading or "",
            chunk_type=row.chunk_type or "text",
            token_count=row.token_count or 0,
            metadata=row.chunk_metadata or {},
        )
        for row in sorted(rows, key=lambda r: r.chunk_index)
    ]


async def _ingest_document(document_id: str) -> dict:
    doc_uuid = _parse_document_id(document_id)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Document).where(Document.id == doc_uuid)
        )
        doc = result.scalar_one_or_none()

    if not doc:
        raise ValueError(f"Document {document_id} not found")

    file_path = storage_service.get_absolute_path(doc.file_path)
    collection_name = settings.user_collection_name(str(doc.user_id))

    await _update_status(document_id, DocumentStatus.PARSING, progress=0.1)
    parsed = parse_document(file_path)

    await _update_status(document_id, DocumentStatus.CHUNKING, progress=0.25)
    chunks = chunk_document(parsed)
    if not chunks:
        raise ValueError(
            "No indexable content extracted from the document. "
            "The file may be empty or in an unsupported layout."
        )

    await _update_status(document_id, DocumentStatus.EMBEDDING, progress=0.45)

    async def _embedding_progress(done: int, total: int) -> None:
        progress = 0.45 + 0.15 * done / total
        await _update_status(document_id, DocumentStatus.EMBEDDING, progress=progress)

    vectors = await embedder_service.embed_chunks(chunks, on_batch=_embedding_progress)

    await _update_status(document_id, DocumentStatus.EMBEDDING, progress=0.60)
    chroma_ids = chroma_service.upsert_chunks(
        collection_name=collection_name,
        chunks=chunks,
        vectors=vectors,
        document_id=document_id,
    )

    await _update_status(document_id, DocumentStatus.EMBEDDING, progress=0.75)

    async with AsyncSessionLocal() as session:
        db_chunks = [
            Chunk(
                document_id=doc_uuid,
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

    await _update_status(document_id, DocumentStatus.ENRICHING, progress=0.85)
    classification = await classify_document(parsed.full_text[:3000])

    entities_status = "pending" if settings.ner_enabled else "skipped"
    doc_type = classification.get("doc_type")
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(Document)
            .where(Document.id == doc_uuid)
            .values(
                page_count=parsed.total_pages,
                doc_type=doc_type.value if hasattr(doc_type, "value") else doc_type,
                doc_metadata={
                    "title": classification.get("title", ""),
                    "author": classification.get("author", ""),
                    "summary": classification.get("summary", ""),
                    "key_dates": classification.get("key_dates", []),
                    "total_chunks": len(chunks),
                    "total_entities": 0,
                    "entities_status": entities_status,
                },
            )
        )
        await session.commit()

    await _update_status(document_id, DocumentStatus.READY, progress=1.0)
    log.info("pipeline.core_complete", document_id=document_id, chunks=len(chunks))

    if settings.ner_enabled:
        _spawn_background_enrich(document_id)

    return {"document_id": document_id, "chunks": len(chunks), "status": "ready"}


def _spawn_background_enrich(document_id: str) -> None:
    """Run NER outside Celery so the worker can pick up the next ingest job."""

    def _worker() -> None:
        with _enrich_thread_slot:
            try:
                _enrich_document_entities(document_id)
            except Exception as exc:
                log.error(
                    "pipeline.enrich_background_failed",
                    document_id=document_id,
                    error=str(exc),
                )

    thread = threading.Thread(
        target=_worker,
        name=f"enrich-{document_id[:8]}",
        daemon=True,
    )
    thread.start()
    log.info("pipeline.enrich_background_started", document_id=document_id)


def _enrich_document_entities(document_id: str) -> dict:
    """Sync DB + isolated asyncio for OpenAI NER (safe in background threads)."""
    doc_uuid = _parse_document_id(document_id)

    with SyncSessionLocal() as session:
        doc = session.execute(
            select(Document).where(Document.id == doc_uuid)
        ).scalar_one_or_none()
        if not doc:
            log.warning("pipeline.enrich_document_missing", document_id=document_id)
            return {"document_id": document_id, "status": "skipped", "reason": "not found"}

        db_chunks = session.execute(
            select(Chunk).where(Chunk.document_id == doc_uuid)
        ).scalars().all()
        doc_metadata = dict(doc.doc_metadata or {})

    if not db_chunks:
        log.warning("pipeline.enrich_no_chunks", document_id=document_id)
        _set_entities_status_sync(doc_uuid, "complete", total_entities=0)
        return {"document_id": document_id, "entities": 0, "status": "complete"}

    chunks = _chunks_from_db(db_chunks)
    log.info("pipeline.enrich_start", document_id=document_id, chunks=len(chunks))

    try:
        all_entities = asyncio.run(extract_entities_all_chunks(chunks))
    except Exception as exc:
        log.error("pipeline.enrich_failed", document_id=document_id, error=str(exc))
        _set_entities_status_sync(doc_uuid, "failed", error=str(exc))
        raise

    with SyncSessionLocal() as session:
        session.execute(delete(Entity).where(Entity.document_id == doc_uuid))
        session.add_all(
            [
                Entity(
                    document_id=doc_uuid,
                    entity_type=e.get("type"),
                    entity_text=e.get("text"),
                    page_number=e.get("page_number"),
                    confidence=e.get("confidence", 0.8),
                )
                for e in all_entities
            ]
        )

        meta = dict(doc_metadata)
        meta["total_entities"] = len(all_entities)
        meta["entities_status"] = "complete"
        meta.pop("entities_error", None)

        session.execute(
            update(Document).where(Document.id == doc_uuid).values(doc_metadata=meta)
        )
        session.commit()

    log.info(
        "pipeline.enrich_complete",
        document_id=document_id,
        entities=len(all_entities),
    )
    return {
        "document_id": document_id,
        "entities": len(all_entities),
        "status": "complete",
    }


def _set_entities_status_sync(
    doc_uuid: uuid.UUID,
    status: str,
    total_entities: int | None = None,
    error: str | None = None,
) -> None:
    with SyncSessionLocal() as session:
        doc = session.execute(
            select(Document).where(Document.id == doc_uuid)
        ).scalar_one_or_none()
        if not doc:
            return
        meta = dict(doc.doc_metadata or {})
        meta["entities_status"] = status
        if total_entities is not None:
            meta["total_entities"] = total_entities
        if error:
            meta["entities_error"] = error
        else:
            meta.pop("entities_error", None)
        session.execute(
            update(Document).where(Document.id == doc_uuid).values(doc_metadata=meta)
        )
        session.commit()


def _run_async(coro, *, dispose_engine: bool = True):
    """Run coroutine on a fresh event loop (safe for Celery workers)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        if dispose_engine:
            loop.run_until_complete(engine.dispose())
        loop.close()
        asyncio.set_event_loop(None)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def task_ingest_document(self, document_id: str) -> dict:
    log.info("pipeline.start", document_id=document_id)
    try:
        return _run_async(_ingest_document(document_id))
    except ValueError as exc:
        if "not found" in str(exc).lower():
            log.warning("pipeline.document_missing", document_id=document_id, error=str(exc))
            return {"document_id": document_id, "status": "skipped", "reason": str(exc)}
        log.error("pipeline.failed", document_id=document_id, error=str(exc))
        try:
            _run_async(_update_status(document_id, DocumentStatus.FAILED, error=str(exc)))
        except Exception as update_exc:
            log.error("pipeline.failed_status_update", error=str(update_exc))
        raise self.retry(exc=exc)
    except Exception as exc:
        log.error("pipeline.failed", document_id=document_id, error=str(exc))
        try:
            _run_async(_update_status(document_id, DocumentStatus.FAILED, error=str(exc)))
        except Exception as update_exc:
            log.error("pipeline.failed_status_update", error=str(update_exc))
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def task_enrich_entities(self, document_id: str) -> dict:
    """Manual / retry path — normal ingest uses a background thread instead."""
    log.info("pipeline.enrich_queued", document_id=document_id)
    try:
        return _enrich_document_entities(document_id)
    except Exception as exc:
        log.error("pipeline.enrich_task_failed", document_id=document_id, error=str(exc))
        raise self.retry(exc=exc)
