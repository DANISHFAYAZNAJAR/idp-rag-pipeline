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


def rerank(query: str, chunks: list[dict], top_n: int | None = None) -> list[dict]:
    if not settings.enable_reranker or not chunks:
        return chunks[: top_n or settings.reranker_top_n]

    top_n = top_n or settings.reranker_top_n
    model = get_reranker()
    pairs = [(query, c["text"]) for c in chunks]

    try:
        scores = model.predict(pairs).tolist()
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = score

        ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        log.info(
            "reranker.complete",
            input=len(chunks),
            output=top_n,
            top_score=round(ranked[0]["rerank_score"], 3) if ranked else 0,
        )
        return ranked[:top_n]
    except Exception as e:
        log.error("reranker.failed", error=str(e))
        return chunks[:top_n]
