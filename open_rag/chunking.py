"""Fixed-width chunks with source attribution."""

from dataclasses import dataclass
from pathlib import Path
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .documents import LoadedDocument


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    filename: str
    page: int | None
    digest: str
    ordinal: int


_splitter = RecursiveCharacterTextSplitter(
    separators=["\n", ". ", " ", ""], chunk_size=500, chunk_overlap=50
)


def split_document(document: LoadedDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    for segment in document.segments:
        for paragraph in re.split(r"\n\s*\n", segment.text):
            if not paragraph.strip():
                continue
            for text in _splitter.split_text(paragraph.strip()):
                if text.strip():
                    chunks.append(
                        Chunk(text, document.source, Path(document.source).name, segment.page, document.digest, len(chunks))
                    )
    return chunks
