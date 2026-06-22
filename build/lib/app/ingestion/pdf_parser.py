from pathlib import Path

import pdfplumber
import structlog
from unstructured.partition.pdf import partition_pdf

from app.ingestion.parsed_document import ParsedDocument, ParsedPage

log = structlog.get_logger()


def parse_pdf(file_path: Path) -> ParsedDocument:
    log.info("pdf_parser.start", path=str(file_path))
    pages: list[ParsedPage] = []
    metadata: dict = {}

    try:
        with pdfplumber.open(file_path) as pdf:
            metadata = {
                "title": pdf.metadata.get("Title", "") if pdf.metadata else "",
                "author": pdf.metadata.get("Author", "") if pdf.metadata else "",
                "creator": pdf.metadata.get("Creator", "") if pdf.metadata else "",
                "total_pages": len(pdf.pages),
            }
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                tables_md = []
                for table in page.extract_tables():
                    if table:
                        rows = [
                            " | ".join(str(c or "") for c in row) for row in table
                        ]
                        tables_md.append("\n".join(rows))

                if len(text.strip()) < 50:
                    text = _fallback_unstructured(file_path, i)

                pages.append(
                    ParsedPage(
                        page_number=i,
                        text=text.strip(),
                        tables=tables_md,
                        has_images=len(page.images) > 0,
                    )
                )

    except Exception as e:
        log.error("pdf_parser.pdfplumber_failed", error=str(e), path=str(file_path))
        pages = _full_unstructured_parse(file_path)
        metadata = {"total_pages": len(pages)}

    log.info("pdf_parser.complete", total_pages=len(pages))
    return ParsedDocument(pages=pages, total_pages=len(pages), metadata=metadata)


def _fallback_unstructured(file_path: Path, page_number: int) -> str:
    try:
        elements = partition_pdf(
            filename=str(file_path),
            strategy="hi_res",
            include_page_breaks=True,
        )
        page_texts = [
            e.text
            for e in elements
            if hasattr(e, "metadata")
            and getattr(e.metadata, "page_number", None) == page_number
            and e.text
        ]
        return "\n".join(page_texts)
    except Exception as e:
        log.warning(
            "pdf_parser.unstructured_fallback_failed", page=page_number, error=str(e)
        )
        return ""


def _full_unstructured_parse(file_path: Path) -> list[ParsedPage]:
    elements = partition_pdf(filename=str(file_path), strategy="fast")
    page_map: dict[int, list[str]] = {}
    for el in elements:
        pn = getattr(getattr(el, "metadata", None), "page_number", 1) or 1
        page_map.setdefault(pn, []).append(el.text or "")
    return [
        ParsedPage(page_number=pn, text="\n".join(texts))
        for pn, texts in sorted(page_map.items())
    ]
