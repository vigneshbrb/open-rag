from pathlib import Path

from langchain_core.messages import AIMessage

from test_index import FixedEmbeddings


class AnswerModel:
    def invoke(self, messages):
        return AIMessage(content="Alpha is documented.")


def prepare_index(tmp_path: Path, monkeypatch) -> None:
    from open_rag import index as index_module
    from open_rag import providers

    source = tmp_path / "alpha.txt"
    source.write_text("Alpha evidence", encoding="utf-8")
    index_module.DocumentIndex(tmp_path / "chroma_db", FixedEmbeddings()).index_file(source)
    monkeypatch.setattr(index_module, "HuggingFaceEmbeddings", lambda **kwargs: FixedEmbeddings())
    monkeypatch.setattr(providers, "make_model", lambda provider: AnswerModel())
    monkeypatch.chdir(tmp_path)


def test_ask_prints_answer_on_stdout_and_trace_on_stderr(tmp_path: Path, monkeypatch, capsys) -> None:
    from open_rag.cli import main

    prepare_index(tmp_path, monkeypatch)
    assert main(["ask", "alpha question"]) == 0
    output = capsys.readouterr()
    assert output.out == f"Alpha is documented.\n\nSources supplied:\n- {tmp_path / 'alpha.txt'}\n"
    assert "[Retrieval]\n" in output.err
    assert "full prompt:" in output.err
    assert "[Retrieval]" not in output.out


def test_chat_handles_multiple_questions_and_exit(tmp_path: Path, monkeypatch, capsys) -> None:
    from open_rag.cli import main

    prepare_index(tmp_path, monkeypatch)
    answers = iter(["alpha question", "unrelated", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert main(["chat"]) == 0
    output = capsys.readouterr()
    assert "Alpha is documented." in output.out
    assert "I cannot find the answer in the document." in output.out
    assert output.err.count("[Question]") == 2


def test_ask_reports_empty_index(tmp_path: Path, monkeypatch, capsys) -> None:
    from open_rag.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["ask", "alpha question"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "No documents indexed" in output.err


def test_ingest_command_upgrades_legacy_index_without_editing_source(tmp_path: Path, monkeypatch, capsys) -> None:
    from open_rag import index as index_module
    from open_rag.cli import main

    source = tmp_path / "notes.txt"
    source.write_text("Alpha evidence", encoding="utf-8")
    original = source.read_bytes()
    index = index_module.DocumentIndex(tmp_path / "chroma_db", FixedEmbeddings())
    index.collection.add(
        ids=["legacy-id"],
        documents=["Alpha evidence"],
        embeddings=[FixedEmbeddings().embed_query("Alpha evidence")],
        metadatas=[{"source": str(source.resolve()), "filename": source.name, "content_hash": "old", "ordinal": 0}],
    )
    monkeypatch.setattr(index_module, "HuggingFaceEmbeddings", lambda **kwargs: FixedEmbeddings())
    monkeypatch.chdir(tmp_path)

    assert main(["ingest", str(source)]) == 0
    assert capsys.readouterr().out == f"indexed: {source}\n"
    assert source.read_bytes() == original
    stored = index.collection.get(include=["metadatas"])
    assert len(stored["ids"]) == 1
    assert stored["metadatas"][0]["chunk_format"] == 2
