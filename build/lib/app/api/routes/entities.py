import uuid
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db.base import get_db
from app.db.models import Document, Entity, User

router = APIRouter(prefix="/entities", tags=["entities"])


@router.get("/{document_id}")
async def get_document_entities(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc_result = await db.execute(
        select(Document).where(Document.id == document_id, Document.user_id == current_user.id)
    )
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    result = await db.execute(select(Entity).where(Entity.document_id == document_id))
    entities = result.scalars().all()

    grouped: dict[str, list] = defaultdict(list)
    for e in entities:
        grouped[e.entity_type].append(
            {"text": e.entity_text, "page": e.page_number, "confidence": e.confidence}
        )

    meta = doc.doc_metadata or {}
    return {
        "document_id": str(document_id),
        "total_entities": len(entities),
        "entities_status": meta.get("entities_status"),
        "entities": grouped,
    }

