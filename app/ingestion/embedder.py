import structlog
from collections.abc import Awaitable, Callable
from openai import AsyncOpenAI

from app.core.config import settings
from app.ingestion.chunker import ParsedChunk

log = structlog.get_logger()


class EmbedderService:
    """OpenAI embeddings via HTTP — safe for Celery workers (no LangChain/tiktoken fork issues)."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def embed_chunks(
        self,
        chunks: list[ParsedChunk],
        on_batch: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> list[list[float]]:
        texts = [c.text for c in chunks]
        all_embeddings: list[list[float]] = []
        batch_size = settings.embedding_batch_size
        total_batches = max(1, (len(texts) + batch_size - 1) // batch_size)

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            log.info(
                "embedder.batch",
                batch_num=batch_num,
                batch_size=len(batch),
            )
            response = await self.client.embeddings.create(
                model=settings.openai_embedding_model,
                input=batch,
            )
            sorted_data = sorted(response.data, key=lambda d: d.index)
            all_embeddings.extend([d.embedding for d in sorted_data])
            if on_batch:
                await on_batch(batch_num, total_batches)

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=settings.openai_embedding_model,
            input=query,
        )
        return response.data[0].embedding


embedder_service = EmbedderService()
