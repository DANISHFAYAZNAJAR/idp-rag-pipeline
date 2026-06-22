"""In-memory BM25 index cache keyed by document set + chunk signature."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import structlog

from app.core.config import settings
from app.retrieval.bm25_retriever import BM25Retriever

log = structlog.get_logger()


@dataclass
class _CacheEntry:
    retriever: BM25Retriever
    created_at: float


class BM25IndexCache:
    """Reuse tokenized BM25 indexes across queries for the same document set."""

    def __init__(self, max_entries: int, ttl_seconds: float) -> None:
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._cache: dict[tuple, _CacheEntry] = {}

    @staticmethod
    def _cache_key(document_ids: list[str], chunks: list[dict]) -> tuple:
        doc_key = tuple(sorted(document_ids))
        chroma_ids = tuple(sorted(c.get("chroma_id") or "" for c in chunks))
        return (doc_key, len(chunks), hash(chroma_ids))

    def get(self, document_ids: list[str], chunks: list[dict]) -> BM25Retriever:
        key = self._cache_key(document_ids, chunks)
        now = time.monotonic()

        with self._lock:
            entry = self._cache.get(key)
            if entry and (now - entry.created_at) < self._ttl_seconds:
                log.debug("bm25_cache.hit", documents=len(document_ids), chunks=len(chunks))
                return entry.retriever

            retriever = BM25Retriever(chunks)
            if len(self._cache) >= self._max_entries:
                oldest_key = min(self._cache.items(), key=lambda item: item[1].created_at)[0]
                del self._cache[oldest_key]

            self._cache[key] = _CacheEntry(retriever=retriever, created_at=now)
            log.info("bm25_cache.miss", documents=len(document_ids), chunks=len(chunks))
            return retriever

    def invalidate_documents(self, document_ids: list[str]) -> None:
        if not document_ids:
            return
        targets = set(document_ids)
        with self._lock:
            before = len(self._cache)
            self._cache = {
                key: entry
                for key, entry in self._cache.items()
                if not targets.intersection(key[0])
            }
            removed = before - len(self._cache)
            if removed:
                log.info("bm25_cache.invalidated", documents=document_ids, entries=removed)


bm25_cache = BM25IndexCache(
    max_entries=settings.bm25_cache_max_entries,
    ttl_seconds=settings.bm25_cache_ttl_seconds,
)
