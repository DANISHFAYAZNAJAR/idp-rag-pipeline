import json

import structlog
from openai import AsyncOpenAI

from app.core.config import settings

log = structlog.get_logger()
client = AsyncOpenAI(api_key=settings.openai_api_key)

MULTI_QUERY_PROMPT = """You are an expert at reformulating search queries.
Given a user question, generate 3 different versions that capture different angles.
Respond ONLY with JSON: {"queries": ["...", "...", "..."]}"""

HYDE_PROMPT = """Given the question below, write a short (3-4 sentence) hypothetical
document excerpt that would perfectly answer it. This will be used for semantic search.
Do NOT add any preamble."""


async def expand_query(query: str) -> list[str]:
    try:
        response = await client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": MULTI_QUERY_PROMPT},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        data = json.loads(response.choices[0].message.content)
        variants = data.get("queries", [])
        return [query] + variants[:3]
    except Exception as e:
        log.warning("query_rewriter.expand_failed", error=str(e))
        return [query]


async def generate_hyde_embedding_text(query: str) -> str:
    try:
        response = await client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": HYDE_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        log.warning("query_rewriter.hyde_failed", error=str(e))
        return query
