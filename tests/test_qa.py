from open_rag.index import Hit
from io import StringIO
from pathlib import Path

from test_index import FixedEmbeddings
import pytest


def hit(source: str, page: int | None, text: str) -> Hit:
    return Hit("id", text, source, source.rsplit("/", 1)[-1], page, 0.1, 0.9)


def test_prompt_preserves_exact_system_wording_and_labels_context() -> None:
    from open_rag.qa import build_prompt

    prompt = build_prompt("What happened?", [hit("/docs/report.pdf", 3, "Alpha happened.")])
    assert prompt == (
        "You are a helpful assistant. Answer the user's question using ONLY the provided text context below. "
        "If the context does not contain the answer, say 'I cannot find the answer in the document.' "
        "Do not make up information.Context: [1] /docs/report.pdf, p. 3\nAlpha happened.\n"
        "Question: What happened?"
    )


def test_source_list_deduplicates_locations_and_refusal_has_none() -> None:
    from open_rag.qa import REFUSAL, format_answer

    hits = [
        hit("/docs/report.pdf", 3, "Alpha"),
        hit("/docs/report.pdf", 3, "Beta"),
        hit("/docs/notes.txt", None, "Gamma"),
    ]
    assert format_answer("The answer.", hits) == (
        "The answer.\n\nSources supplied:\n- /docs/report.pdf, p. 3\n- /docs/notes.txt"
    )
    assert format_answer(REFUSAL, hits) == REFUSAL


def test_request_trace_shows_retrieval_prompt_and_answer_without_key(tmp_path: Path, monkeypatch) -> None:
    from langchain_core.messages import AIMessage
    from open_rag.index import DocumentIndex
    from open_rag.qa import answer_question

    source = tmp_path / "alpha.txt"
    source.write_text("Alpha evidence", encoding="utf-8")
    index = DocumentIndex(tmp_path / "chroma_db", FixedEmbeddings())
    index.index_file(source)
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-api-key")
    trace = StringIO()

    class Model:
        def invoke(self, messages):
            assert len(messages) == 1 and messages[0].type == "system"
            assert "Alpha evidence" in messages[0].content
            return AIMessage(content="Alpha is documented.")

    output = answer_question(index, "alpha question", "openai", trace=trace, model=Model())
    assert output.startswith("Alpha is documented.\n\nSources supplied:")
    log = trace.getvalue()
    for term in ("model load", "embedding", "Chroma query", "rank=1", "Alpha evidence", "selected context", "system prompt", "provider=openai", "answer"):
        assert term in log
    assert "secret-test-api-key" not in log


def test_unrelated_request_traces_refusal_without_model_call(tmp_path: Path) -> None:
    from open_rag.index import DocumentIndex
    from open_rag.qa import REFUSAL, answer_question

    source = tmp_path / "alpha.txt"
    source.write_text("Alpha evidence", encoding="utf-8")
    index = DocumentIndex(tmp_path / "chroma_db", FixedEmbeddings())
    index.index_file(source)
    trace = StringIO()

    class UnusedModel:
        def invoke(self, messages):
            raise AssertionError("model should not be called")

    assert answer_question(index, "unrelated", "ollama", trace=trace, model=UnusedModel()) == REFUSAL
    assert "generation skipped" in trace.getvalue()


def test_provider_error_is_clear_and_does_not_echo_credentials(tmp_path: Path, monkeypatch) -> None:
    from open_rag.index import DocumentIndex
    from open_rag.providers import ProviderError
    from open_rag.qa import answer_question

    source = tmp_path / "alpha.txt"
    source.write_text("Alpha evidence", encoding="utf-8")
    index = DocumentIndex(tmp_path / "chroma_db", FixedEmbeddings())
    index.index_file(source)
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-api-key")

    class FailingModel:
        def invoke(self, messages):
            raise RuntimeError("secret-test-api-key leaked from transport")

    trace = StringIO()
    with pytest.raises(ProviderError, match="OpenAI request failed") as error:
        answer_question(index, "alpha question", "openai", trace=trace, model=FailingModel())
    assert "secret-test-api-key" not in str(error.value) + trace.getvalue()


def test_local_generation_disables_optional_langsmith_tracing(tmp_path: Path, monkeypatch) -> None:
    from langchain_core.messages import AIMessage
    from langsmith.run_helpers import get_tracing_context
    from open_rag.index import DocumentIndex
    from open_rag.qa import answer_question

    source = tmp_path / "alpha.txt"
    source.write_text("Alpha evidence", encoding="utf-8")
    index = DocumentIndex(tmp_path / "chroma_db", FixedEmbeddings())
    index.index_file(source)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    class Model:
        def invoke(self, messages):
            assert get_tracing_context()["enabled"] is False
            return AIMessage(content="Alpha is documented.")

    assert "Alpha is documented." in answer_question(index, "alpha question", trace=StringIO(), model=Model())
