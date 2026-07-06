import time
import uuid
from typing import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.weave_setup import weave_attributes
from app.db.base import get_db
from app.db.models import Chunk, Document, DocumentStatus, QuerySession, QueryTurn, User
from app.retrieval.graph import (
    RAGState,
    _build_llm_messages,
    _chunk_page_number,
    node_fuse_and_rerank,
    node_hybrid_retrieve,
    node_rewrite_query,
    rag_graph,
)
from app.schemas.query import Citation, QueryRequest, QueryResponse

log = structlog.get_logger()
router = APIRouter(prefix="/query", tags=["query"])


async def _load_rag_state(
    document_ids: list[uuid.UUID],
    user_id: uuid.UUID,
    db: AsyncSession,
) -> tuple[str, list[dict]]:
    result = await db.execute(
        select(Document).where(Document.id.in_(document_ids), Document.user_id == user_id)
    )
    docs = result.scalars().all()

    if len(docs) != len(document_ids):
        found_ids = {str(d.id) for d in docs}
        missing = [str(d) for d in document_ids if str(d) not in found_ids]
        raise HTTPException(
            status_code=404,
            detail=f"Document(s) not found: {missing}. Use GET /documents to list valid IDs.",
        )

    not_ready = [d for d in docs if d.status != DocumentStatus.READY]
    if not_ready:
        raise HTTPException(
            status_code=409,
            detail=f"Documents not ready: {[str(d.id) for d in not_ready]}",
        )

    collection_name = settings.user_collection_name(str(user_id))

    chunks_result = await db.execute(select(Chunk).where(Chunk.document_id.in_(document_ids)))
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
    if not db_chunks:
        raise HTTPException(
            status_code=409,
            detail=(
                "Selected document(s) have no indexed content. "
                "Delete and re-upload the file, or pick a different document. "
                f"Files: {[d.original_filename for d in docs]}"
            ),
        )
    return collection_name, db_chunks


async def _get_or_create_session(
    db: AsyncSession,
    user_id: uuid.UUID,
    document_ids: list[uuid.UUID],
    session_id: uuid.UUID | None,
) -> tuple[uuid.UUID, list[dict]]:
    if session_id:
        result = await db.execute(
            select(QuerySession).where(
                QuerySession.id == session_id,
                QuerySession.user_id == user_id,
            )
        )
        session = result.scalar_one_or_none()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        sid = session.id
    else:
        session = QuerySession(
            user_id=user_id,
            document_ids=[str(d) for d in document_ids],
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)
        sid = session.id

    turns_result = await db.execute(
        select(QueryTurn)
        .where(QueryTurn.session_id == sid)
        .order_by(QueryTurn.created_at)
    )
    chat_history = [
        {"question": t.question, "answer": t.answer or ""}
        for t in turns_result.scalars().all()
    ]
    return sid, chat_history


def _resolve_chat_history(body: QueryRequest, db_history: list[dict]) -> list[dict]:
    """Prefer client-provided history (UI); fall back to DB session turns."""
    if body.chat_history:
        return body.chat_history
    return db_history


@router.post("", response_model=QueryResponse)
async def query_documents(
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    start_ms = int(time.time() * 1000)

    collection_name, db_chunks = await _load_rag_state(body.document_ids, current_user.id, db)

    session_id, db_history = await _get_or_create_session(
        db, current_user.id, body.document_ids, body.session_id
    )
    chat_history = _resolve_chat_history(body, db_history)

    with weave_attributes({"session_id": str(session_id), "user_id": str(current_user.id)}):
        initial_state: RAGState = {
            "query": body.question,
            "user_id": str(current_user.id),
            "document_ids": [str(d) for d in body.document_ids],
            "collection_name": collection_name,
            "db_chunks": db_chunks,
            "chat_history": chat_history,
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
    collection_name, db_chunks = await _load_rag_state(body.document_ids, current_user.id, db)
    session_id, db_history = await _get_or_create_session(
        db, current_user.id, body.document_ids, body.session_id
    )
    chat_history = _resolve_chat_history(body, db_history)

    async def event_generator() -> AsyncIterator[str]:
        import json

        from app.retrieval.graph import NO_ANSWER_MESSAGE, _build_llm_messages, _chunk_page_number

        try:
            state: RAGState = {
                "query": body.question,
                "user_id": str(current_user.id),
                "document_ids": [str(d) for d in body.document_ids],
                "collection_name": collection_name,
                "db_chunks": db_chunks,
                "chat_history": chat_history,
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

            yield f"data: {json.dumps({'type': 'status', 'status': 'rewriting_query'})}\n\n"
            state.update(await node_rewrite_query(state))

            yield f"data: {json.dumps({'type': 'status', 'status': 'retrieving'})}\n\n"
            state.update(await node_hybrid_retrieve(state))

            yield f"data: {json.dumps({'type': 'status', 'status': 'reranking'})}\n\n"
            state.update(await node_fuse_and_rerank(state))

            yield f"data: {json.dumps({'type': 'debug', 'vector': len(state['vector_results']), 'bm25': len(state['bm25_results']), 'reranked': len(state['reranked_results'])})}\n\n"

            yield f"data: {json.dumps({'type': 'status', 'status': 'generating'})}\n\n"

            if state["low_confidence"]:
                yield f"data: {json.dumps({'type': 'token', 'content': NO_ANSWER_MESSAGE})}\n\n"
            else:
                context_parts = [
                    f"[Page {_chunk_page_number(c)}]\n{c['text']}"
                    for c in state["reranked_results"]
                ]
                context = "\n\n---\n\n".join(context_parts)
                llm = ChatOpenAI(
                    model=settings.openai_chat_model_answer,
                    temperature=0,
                    streaming=True,
                    openai_api_key=settings.openai_api_key,
                )
                messages = _build_llm_messages(state, context)
                token_count = 0
                async for chunk in llm.astream(messages):
                    if chunk.content:
                        token_count += 1
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
                if token_count == 0:
                    yield f"data: {json.dumps({'type': 'token', 'content': NO_ANSWER_MESSAGE})}\n\n"

            citations = [
                {
                    "page": _chunk_page_number(c) if _chunk_page_number(c) != "?" else None,
                    "snippet": c["text"][:150],
                    "score": round(c.get("rerank_score") or c.get("rrf_score", 0), 3),
                }
                for c in state["reranked_results"]
            ]
            yield f"data: {json.dumps({'type': 'citations', 'citations': citations, 'confidence': state['confidence'], 'session_id': str(session_id)})}\n\n"
            yield 'data: {"type": "done"}\n\n'
        except Exception as exc:
            log.error("query.stream_failed", error=str(exc))
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc), 'retry': True})}\n\n"
            yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")

