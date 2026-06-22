"""Reciprocal Rank Fusion for dense vector + BM25 results."""
from collections import defaultdict


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[dict]:
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, dict] = {}

    for rank, result in enumerate(vector_results, start=1):
        cid = result.get("chroma_id") or str(result.get("chunk_index", rank))
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
