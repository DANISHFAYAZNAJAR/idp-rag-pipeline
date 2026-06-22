from pathlib import Path

import structlog
from docx import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.ingestion.parsed_document import ParsedDocument, ParsedPage

log = structlog.get_logger()


def _iter_block_items(parent):
    """Yield paragraphs and tables in document order."""
    parent_elm = parent.element.body
    for child in parent_elm.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def _table_to_markdown(table: Table) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def _is_heading1(paragraph: Paragraph) -> bool:
    style = paragraph.style.name if paragraph.style else ""
    return style == "Heading 1" or style.startswith("Heading 1")


def parse_docx(file_path: Path) -> ParsedDocument:
    log.info("docx_parser.start", path=str(file_path))
    doc = DocxDocument(file_path)

    props = doc.core_properties
    metadata = {
        "title": props.title or "",
        "author": props.author or "",
        "creator": props.author or "",
        "total_pages": 0,
    }

    pages: list[ParsedPage] = []
    current_text: list[str] = []
    current_tables: list[str] = []
    page_number = 1

    def flush_page() -> None:
        nonlocal page_number, current_text, current_tables
        text = "\n\n".join(t for t in current_text if t.strip()).strip()
        if text or current_tables:
            pages.append(
                ParsedPage(
                    page_number=page_number,
                    text=text,
                    tables=list(current_tables),
                )
            )
            page_number += 1
        current_text = []
        current_tables = []

    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            if pages or current_text or current_tables:
                if _is_heading1(block):
                    flush_page()
            current_text.append(text)
        elif isinstance(block, Table):
            md = _table_to_markdown(block)
            if md.strip():
                current_tables.append(md)

    flush_page()

    if not pages:
        pages.append(ParsedPage(page_number=1, text=""))

    metadata["total_pages"] = len(pages)
    log.info("docx_parser.complete", total_pages=len(pages))
    return ParsedDocument(pages=pages, total_pages=len(pages), metadata=metadata)
