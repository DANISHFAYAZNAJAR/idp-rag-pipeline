"""Parse uploaded documents (PDF, DOCX) into a common structure."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.docx_parser import parse_docx
from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.parsed_document import ParsedDocument

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx"})


def is_supported_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def parse_document(file_path: Path) -> ParsedDocument:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path)
    if suffix == ".docx":
        return parse_docx(file_path)
    raise ValueError(
        f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )
