from io import StringIO
from pathlib import Path

from langchain_core.messages import AIMessage

from test_documents import make_pdf
from test_index import FixedEmbeddings


class AnswerModel:
    def invoke(self, messages):
        return AIMessage(content="Beta evidence is in the PDF.")


def test_ten_page_sized_text_and_pdf_persist_refresh_and_trace(tmp_path: Path) -> None:
    from open_rag.index import DocumentIndex
    from open_rag.qa import answer_question

    text_path = tmp_path / "manual.txt"
    text_path.write_text(("Alpha reference material. " * 80 + "\n") * 10, encoding="utf-8")
    pdf_path = tmp_path / "report.pdf"
    make_pdf(pdf_path, "Beta evidence is in the PDF.")
    db = tmp_path / "chroma_db"

    first_run = DocumentIndex(db, FixedEmbeddings())
    first_run.index_file(text_path)
    first_run.index_file(pdf_path)
    initial_count = first_run.count()
    assert initial_count > 10

    restarted = DocumentIndex(db, FixedEmbeddings())
    assert restarted.count() == initial_count
    assert restarted.index_file(text_path) == "unchanged"
    assert restarted.search("alpha question").usable[0].filename == "manual.txt"
    trace = StringIO()
    answer = answer_question(restarted, "beta question", trace=trace, model=AnswerModel())
    assert answer.endswith(f"Sources supplied:\n- {pdf_path}, p. 1")
    for stage in ("model load", "embedding", "Chroma query", "rank=1", "selected context", "full prompt", "provider=ollama", "answer"):
        assert stage in trace.getvalue()

    text_path.write_text("Gamma replacement only.", encoding="utf-8")
    assert restarted.index_file(text_path) == "indexed"
    documents = restarted.collection.get(include=["documents"])["documents"]
    assert "Gamma replacement only." in documents
    assert all("Alpha reference" not in chunk for chunk in documents)
    assert restarted.search("beta question").usable[0].filename == "report.pdf"
