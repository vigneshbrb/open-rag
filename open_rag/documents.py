"""Read source documents before changing the index."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


class DocumentError(ValueError):
    """A document cannot be indexed."""


@dataclass(frozen=True)
class Segment:
    text: str
    page: int | None


@dataclass(frozen=True)
class LoadedDocument:
    source: str
    digest: str
    segments: tuple[Segment, ...]


def load_document(path: str | Path) -> LoadedDocument:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() not in {".txt", ".pdf"}:
        raise DocumentError(f"{source}: unsupported file type; use .txt or .pdf")
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise DocumentError(f"{source}: cannot read file: {exc.strerror or exc}") from exc

    if source.suffix.lower() == ".txt":
        try:
            segments = (Segment(data.decode("utf-8"), None),)
        except UnicodeDecodeError as exc:
            raise DocumentError(f"{source}: text must be UTF-8") from exc
    else:
        try:
            pdf = PdfReader(BytesIO(data))
            segments = tuple(Segment(page.extract_text() or "", number) for number, page in enumerate(pdf.pages, 1))
        except Exception as exc:
            raise DocumentError(f"{source}: cannot extract PDF text: {exc}") from exc

    segments = tuple(segment for segment in segments if segment.text.strip())
    if not segments:
        raise DocumentError(f"{source}: no extractable text")
    return LoadedDocument(str(source), sha256(data).hexdigest(), segments)
