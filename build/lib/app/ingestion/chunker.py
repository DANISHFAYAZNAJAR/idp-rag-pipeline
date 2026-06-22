from dataclasses import dataclass

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.ingestion.parsed_document import ParsedDocument

log = structlog.get_logger()


@dataclass
class ParsedChunk:
    text: str
    page_number: int
    chunk_index: int
    section_heading: str
    chunk_type: str
    token_count: int
    metadata: dict


def chunk_document(parsed: ParsedDocument) -> list[ParsedChunk]:
    """
    Chunking strategy (fork-safe for Celery workers):
    - Tables → one chunk each
    - Short docs (≤10 pages) → recursive splitter
    - Long docs → paragraph-aware packing (semantic boundaries, no ML libs)
    """
    chunks: list[ParsedChunk] = []
    index = 0

    for page in parsed.pages:
        for table_md in page.tables:
            if table_md.strip():
                chunks.append(
                    ParsedChunk(
                        text=f"[TABLE]\n{table_md}",
                        page_number=page.page_number,
                        chunk_index=index,
                        section_heading="",
                        chunk_type="table",
                        token_count=_estimate_tokens(table_md),
                        metadata={"page": page.page_number, "type": "table"},
                    )
                )
                index += 1

    text_chunks = _chunk_text(parsed, start_index=index)
    chunks.extend(text_chunks)

    log.info("chunker.complete", total_chunks=len(chunks))
    return chunks[: settings.max_chunks_per_doc]


def _chunk_text(parsed: ParsedDocument, start_index: int) -> list[ParsedChunk]:
    segments: list[tuple[int, str]] = [
        (p.page_number, p.text) for p in parsed.pages if p.text.strip()
    ]

    if parsed.total_pages > 10:
        log.info("chunker.strategy", strategy="paragraph")
        return _paragraph_chunk(segments, start_index)

    log.info("chunker.strategy", strategy="recursive")
    return _recursive_chunk(segments, start_index)


def _paragraph_chunk(
    segments: list[tuple[int, str]], start_index: int
) -> list[ParsedChunk]:
    """
    For long docs: split on paragraph boundaries, pack into ~chunk_size blocks.
    Avoids SemanticChunker which SIGSEGVs in Celery fork workers.
    """
    chunks: list[ParsedChunk] = []
    idx = start_index
    target = settings.chunk_size
    overlap = settings.chunk_overlap

    for page_num, text in segments:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            continue

        buffer = ""
        for para in paragraphs:
            candidate = f"{buffer}\n\n{para}".strip() if buffer else para
            if len(candidate) <= target:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(_make_chunk(buffer, page_num, idx))
                    idx += 1
                    # overlap: carry trailing text from previous chunk
                    if overlap and len(buffer) > overlap:
                        buffer = buffer[-overlap:] + "\n\n" + para
                    else:
                        buffer = para
                else:
                    # single paragraph exceeds target — fall back to recursive
                    for t in _split_text_recursive(para):
                        chunks.append(_make_chunk(t, page_num, idx))
                        idx += 1
                    buffer = ""

        if buffer:
            chunks.append(_make_chunk(buffer, page_num, idx))
            idx += 1

    return chunks


def _recursive_chunk(
    segments: list[tuple[int, str]], start_index: int
) -> list[ParsedChunk]:
    chunks: list[ParsedChunk] = []
    idx = start_index

    for page_num, text in segments:
        for t in _split_text_recursive(text):
            chunks.append(_make_chunk(t, page_num, idx))
            idx += 1

    return chunks


def _split_text_recursive(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [t for t in splitter.split_text(text) if t.strip()]


def _make_chunk(text: str, page_num: int, idx: int) -> ParsedChunk:
    return ParsedChunk(
        text=text,
        page_number=page_num,
        chunk_index=idx,
        section_heading=_detect_heading(text),
        chunk_type="text",
        token_count=_estimate_tokens(text),
        metadata={"page": page_num, "type": "text"},
    )


def _detect_heading(text: str) -> str:
    first_line = text.strip().split("\n")[0].strip()
    if len(first_line) < 100 and first_line.istitle():
        return first_line
    return ""


def _estimate_tokens(text: str) -> int:
    return len(text) // 4
