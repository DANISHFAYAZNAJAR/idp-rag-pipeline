"""
LangGraph RAG pipeline:
  rewrite_query → hybrid_retrieve → fuse_and_rerank → generate_answer

Weave is required (per project decision) and captures node traces.
"""

from __future__ import annotations

from typing import TypedDict

import structlog
import weave
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
    user_id: str
    document_ids: list[str]
    collection_name: str
    db_chunks: list[dict]
    chat_history: list[dict]
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

Use ONLY the excerpts below to answer the question. Do not use outside knowledge.

Structure your response as:
1. **Direct answer** — one or two sentences that directly address the question.
2. **Explanation** — walk through the evidence: what each relevant excerpt shows, how the pieces connect, and why they support your answer.
3. **Key details** — numbers, dates, and names quoted exactly from the excerpts.
4. **Gaps** — if the excerpts do not contain enough information, say so explicitly.

Rules:
- Every factual claim must cite the source using [Page X] format.
- Do not invent facts not present in the excerpts.
- Prefer clear prose over bullet dumps.

Context:
{context}"""


def _chunk_page_number(chunk: dict) -> str | int:
    metadata = chunk.get("metadata") or {}
    return metadata.get("page_number") or chunk.get("page_number") or "?"


def _retrieval_query(state: RAGState) -> str:
    """Expand short follow-ups using recent conversation."""
    history = state.get("chat_history") or []
    if not history:
        return state["query"]
    lines = []
    for turn in history[-3:]:
        lines.append(f"User: {turn.get('question', '')}")
        if turn.get("answer"):
            lines.append(f"Assistant: {turn['answer'][:400]}")
    lines.append(f"User: {state['query']}")
    return "\n".join(lines)


def _build_llm_messages(state: RAGState, context: str) -> list[dict]:
    messages = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT.format(context=context)}]
    for turn in state.get("chat_history") or []:
        messages.append({"role": "user", "content": turn.get("question", "")})
        if turn.get("answer"):
            messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": state["query"]})
    return messages


@weave.op()
async def node_rewrite_query(state: RAGState) -> dict:
    search_query = _retrieval_query(state)
    log.info("rag.rewrite_query", query=state["query"])
    expanded = await expand_query(search_query)
    hyde_text = await generate_hyde_embedding_text(search_query)
    return {"expanded_queries": expanded, "hyde_text": hyde_text}


@weave.op()
async def node_hybrid_retrieve(state: RAGState) -> dict:
    import asyncio

    async def vector_search_one(text: str) -> list[dict]:
        vec = await embedder_service.embed_query(text)
        return chroma_service.search(
            collection_name=state["collection_name"],
            query_vector=vec,
            n_results=settings.retrieval_top_k,
            document_ids=state["document_ids"],
        )

    search_texts = [state["hyde_text"]] + state["expanded_queries"]
    all_vector_nested = await asyncio.gather(*[vector_search_one(t) for t in search_texts])

    seen: dict[str, dict] = {}
    for results in all_vector_nested:
        for r in results:
            cid = r["chroma_id"]
            if cid not in seen or r["score"] > seen[cid]["score"]:
                seen[cid] = r
    vector_results = list(seen.values())

    bm25_retriever = BM25Retriever(state["db_chunks"])
    bm25_results = bm25_retriever.search(_retrieval_query(state), top_k=settings.retrieval_top_k)

    log.info("rag.retrieved", vector=len(vector_results), bm25=len(bm25_results))
    return {"vector_results": vector_results, "bm25_results": bm25_results}


@weave.op()
async def node_fuse_and_rerank(state: RAGState) -> dict:
    fused = reciprocal_rank_fusion(state["vector_results"], state["bm25_results"])
    reranked = rerank(state["query"], fused, top_n=settings.reranker_top_n)

    scores = [r.get("rerank_score") or r.get("rrf_score", 0) for r in reranked]
    confidence = sum(scores) / len(scores) if scores else -999.0
    # Cross-encoder scores are logits (often negative). Only block when nothing retrieved.
    low_confidence = len(reranked) == 0

    log.info(
        "rag.fused",
        fused=len(fused),
        reranked=len(reranked),
        confidence=round(confidence, 3),
    )
    return {
        "fused_results": fused,
        "reranked_results": reranked,
        "confidence": confidence,
        "low_confidence": low_confidence,
    }


@weave.op()
async def node_generate_answer(state: RAGState) -> dict:
    if state["low_confidence"]:
        return {
            "answer": "I don't have enough relevant information in the provided documents to answer this question accurately.",
            "citations": [],
        }

    context_parts: list[str] = []
    for chunk in state["reranked_results"]:
        page = _chunk_page_number(chunk)
        context_parts.append(f"[Page {page}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    llm = ChatOpenAI(
        model=settings.openai_chat_model_answer,
        temperature=0,
        streaming=False,
        openai_api_key=settings.openai_api_key,
    )

    messages = _build_llm_messages(state, context)

    response = await llm.ainvoke(messages)
    answer = response.content

    citations = [
        {
            "chroma_id": c.get("chroma_id"),
            "page": _chunk_page_number(c) if _chunk_page_number(c) != "?" else None,
            "snippet": c["text"][:200],
            "score": round(c.get("rerank_score") or c.get("rrf_score", 0), 3),
        }
        for c in state["reranked_results"]
    ]

    return {"answer": answer, "citations": citations}


def build_rag_graph():
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

