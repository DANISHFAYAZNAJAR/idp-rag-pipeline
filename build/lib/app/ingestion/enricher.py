import asyncio
import json
from typing import Any

import structlog
from openai import AsyncOpenAI

from app.core.config import settings
from app.db.models import DocumentType
from app.ingestion.chunker import ParsedChunk

log = structlog.get_logger()
client = AsyncOpenAI(api_key=settings.openai_api_key)

CLASSIFY_PROMPT = """You are a document classifier. Given the first 2000 characters of a document,
classify it into one of: invoice, contract, report, research, manual, unknown.
Also extract: title, author (if visible), key_dates (list), summary (2 sentences).

Respond ONLY with valid JSON:
{
  "doc_type": "...",
  "title": "...",
  "author": "...",
  "key_dates": [],
  "summary": "..."
}"""

NER_BATCH_PROMPT = """Extract named entities from each numbered text block below.
Return ONLY valid JSON:
{
  "blocks": [
    {
      "block_id": 0,
      "entities": [
        {"type": "PERSON|ORG|DATE|MONEY|LOCATION|PRODUCT|LAW", "text": "...", "confidence": 0.0-1.0}
      ]
    }
  ]
}
Include all entity types found. Deduplicate within each block."""


async def classify_document(text_sample: str) -> dict[str, Any]:
    try:
        response = await client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user", "content": text_sample[:2000]},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        doc_type_str = result.get("doc_type", "unknown").lower()
        if doc_type_str in DocumentType._value2member_map_:
            result["doc_type"] = DocumentType(doc_type_str)
        else:
            result["doc_type"] = DocumentType.UNKNOWN
        return result
    except Exception as e:
        log.error("enricher.classify_failed", error=str(e))
        return {"doc_type": DocumentType.UNKNOWN}


async def extract_entities_batch(
    chunk_batch: list[ParsedChunk],
) -> list[dict[str, Any]]:
    """LLM NER on a batch of chunks — best online approach without local models."""
    if not chunk_batch:
        return []

    blocks_text = "\n\n".join(
        f"--- Block {i} (page {c.page_number}) ---\n{c.text[:1500]}"
        for i, c in enumerate(chunk_batch)
    )

    try:
        response = await client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": NER_BATCH_PROMPT},
                {"role": "user", "content": blocks_text},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        all_entities: list[dict[str, Any]] = []

        for block in result.get("blocks", []):
            block_id = block.get("block_id", 0)
            page_number = (
                chunk_batch[block_id].page_number
                if block_id < len(chunk_batch)
                else None
            )
            for entity in block.get("entities", []):
                all_entities.append(
                    {
                        "type": entity.get("type", "UNKNOWN"),
                        "text": entity.get("text", ""),
                        "confidence": entity.get("confidence", 0.8),
                        "page_number": page_number,
                    }
                )
        return all_entities
    except Exception as e:
        log.warning("enricher.ner_batch_failed", error=str(e))
        return []


async def extract_entities_all_chunks(
    chunks: list[ParsedChunk],
) -> list[dict[str, Any]]:
    """Run NER on every chunk, batched and parallelized."""
    batch_size = settings.ner_batch_size
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]

    if not batches:
        return []

    sem = asyncio.Semaphore(settings.ner_concurrency)

    async def run_batch(batch_idx: int, batch: list[ParsedChunk]) -> list[dict[str, Any]]:
        async with sem:
            entities = await extract_entities_batch(batch)
            log.info(
                "enricher.ner_batch",
                batch=batch_idx,
                entities_found=len(entities),
            )
            return entities

    results = await asyncio.gather(
        *[run_batch(i, batch) for i, batch in enumerate(batches)]
    )
    all_entities: list[dict[str, Any]] = []
    for batch_entities in results:
        all_entities.extend(batch_entities)

    return _deduplicate_entities(all_entities)


def _deduplicate_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, int | None]] = set()
    unique: list[dict[str, Any]] = []
    for e in entities:
        key = (e.get("type", ""), e.get("text", "").lower(), e.get("page_number"))
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique
