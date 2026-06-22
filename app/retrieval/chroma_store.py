import uuid
from pathlib import Path

import chromadb
import structlog
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.ingestion.chunker import ParsedChunk

log = structlog.get_logger()


class ChromaService:
    def __init__(self):
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def get_or_create_collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[ParsedChunk],
        vectors: list[list[float]],
        document_id: str,
    ) -> list[str]:
        collection = self.get_or_create_collection(collection_name)

        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "chunk_type": c.chunk_type,
                "section_heading": c.section_heading or "",
            }
            for c in chunks
        ]

        batch_size = 500
        for i in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=vectors[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )
            log.info(
                "chroma.upserted",
                batch=i // batch_size,
                count=len(ids[i : i + batch_size]),
            )

        return ids

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        n_results: int = 20,
        document_ids: list[str] | None = None,
    ) -> list[dict]:
        collection = self.get_or_create_collection(collection_name)
        count = collection.count()
        if count == 0:
            return []

        where = None
        if document_ids:
            if len(document_ids) == 1:
                where = {"document_id": document_ids[0]}
            else:
                where = {"document_id": {"$in": document_ids}}

        try:
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=min(n_results, count),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            log.error(
                "chroma.query_failed",
                collection=collection_name,
                document_ids=document_ids,
                error=str(e),
            )
            return []
        if not results["ids"] or not results["ids"][0]:
            return []

        return [
            {
                "chroma_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 - results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]

    def delete_by_document(self, collection_name: str, document_id: str) -> None:
        try:
            collection = self.get_or_create_collection(collection_name)
            collection.delete(where={"document_id": document_id})
            log.info("chroma.deleted_document", collection=collection_name, document_id=document_id)
        except Exception as e:
            log.warning(
                "chroma.delete_failed",
                collection=collection_name,
                document_id=document_id,
                error=str(e),
            )

    def get_document_embeddings(self, document_id: str) -> dict | None:
        """Load all chunk embeddings for a document (searches user collections)."""
        for collection in self.client.list_collections():
            try:
                data = collection.get(
                    where={"document_id": document_id},
                    include=["embeddings", "documents", "metadatas"],
                )
            except Exception as e:
                log.warning(
                    "chroma.get_embeddings_failed",
                    collection=collection.name,
                    error=str(e),
                )
                continue
            if data.get("ids"):
                return {
                    "collection_name": collection.name,
                    "ids": data["ids"],
                    "embeddings": data["embeddings"],
                    "documents": data["documents"],
                    "metadatas": data["metadatas"],
                }
        return None


chroma_service = ChromaService()
