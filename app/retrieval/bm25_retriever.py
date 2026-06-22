from rank_bm25 import BM25Okapi


class BM25Retriever:
    """In-memory BM25 over chunk dicts loaded from PostgreSQL."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.bm25 = None
        if chunks:
            tokenized = [c["text"].lower().split() for c in chunks]
            self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if not self.bm25:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                chunk = self.chunks[idx].copy()
                chunk["bm25_score"] = float(scores[idx])
                chunk["bm25_rank"] = rank + 1
                results.append(chunk)
        return results
