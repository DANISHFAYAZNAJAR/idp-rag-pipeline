from dataclasses import dataclass, field


@dataclass
class ParsedPage:
    page_number: int
    text: str
    tables: list[str] = field(default_factory=list)
    has_images: bool = False


@dataclass
class ParsedDocument:
    pages: list[ParsedPage]
    total_pages: int
    metadata: dict

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())
