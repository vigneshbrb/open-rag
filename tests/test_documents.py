from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def make_pdf(path: Path, text: str | None) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    if text:
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


def test_reads_txt_and_pdf_with_page_metadata(tmp_path: Path) -> None:
    from open_rag.documents import load_document

    txt = tmp_path / "notes.txt"
    txt.write_text("Alpha facts", encoding="utf-8")
    pdf = tmp_path / "facts.pdf"
    make_pdf(pdf, "Beta facts")

    loaded_txt = load_document(txt)
    loaded_pdf = load_document(pdf)
    assert [(s.text, s.page) for s in loaded_txt.segments] == [("Alpha facts", None)]
    assert [(s.text.strip(), s.page) for s in loaded_pdf.segments] == [("Beta facts", 1)]


@pytest.mark.parametrize("name", ["missing.txt", "notes.csv"])
def test_rejects_invalid_path_or_extension(tmp_path: Path, name: str) -> None:
    from open_rag.documents import DocumentError, load_document

    with pytest.raises(DocumentError, match=name):
        load_document(tmp_path / name)


def test_rejects_pdf_with_no_extractable_text(tmp_path: Path) -> None:
    from open_rag.documents import DocumentError, load_document

    pdf = tmp_path / "scan.pdf"
    make_pdf(pdf, None)
    with pytest.raises(DocumentError, match="no extractable text"):
        load_document(pdf)


def test_rejects_invalid_utf8(tmp_path: Path) -> None:
    from open_rag.documents import DocumentError, load_document

    txt = tmp_path / "bad.txt"
    txt.write_bytes(b"\xff")
    with pytest.raises(DocumentError, match="UTF-8"):
        load_document(txt)
