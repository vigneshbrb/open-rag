from pathlib import Path

import pytest

from open_rag.documents import DocumentError


class FixedEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        values = [0.0] * 384
        values[0 if "alpha" in text.lower() else 1 if "beta" in text.lower() else 2] = 1.0
        return values


def test_persists_multiple_documents_without_duplicate_reindex(tmp_path: Path) -> None:
    from open_rag.index import DocumentIndex

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("Alpha evidence", encoding="utf-8")
    second.write_text("Beta evidence", encoding="utf-8")
    index_path = tmp_path / "chroma_db"
    index = DocumentIndex(index_path, FixedEmbeddings())
    assert index.index_file(first) == "indexed"
    assert index.index_file(second) == "indexed"
    assert index.count() == 2
    assert index.index_file(first) == "unchanged"
    assert index.count() == 2

    reopened = DocumentIndex(index_path, FixedEmbeddings())
    assert reopened.count() == 2
    assert {item["source"] for item in reopened.collection.get()["metadatas"]} == {
        str(first.resolve()), str(second.resolve())
    }
    assert len(reopened.collection.get(include=["embeddings"])["embeddings"][0]) == 384


def test_changed_file_replaces_old_chunks_but_invalid_file_preserves_them(tmp_path: Path) -> None:
    from open_rag.index import DocumentIndex

    source = tmp_path / "notes.txt"
    source.write_text("Alpha evidence", encoding="utf-8")
    index = DocumentIndex(tmp_path / "chroma_db", FixedEmbeddings())
    index.index_file(source)
    source.write_text("Beta replacement", encoding="utf-8")
    assert index.index_file(source) == "indexed"
    assert index.collection.get(include=["documents"])["documents"] == ["Beta replacement"]

    source.write_bytes(b"\xff")
    with pytest.raises(DocumentError):
        index.index_file(source)
    assert index.collection.get(include=["documents"])["documents"] == ["Beta replacement"]


def test_reindex_reconciles_stale_version(tmp_path: Path) -> None:
    from open_rag.index import DocumentIndex

    source = tmp_path / "notes.txt"
    source.write_text("Alpha evidence", encoding="utf-8")
    index = DocumentIndex(tmp_path / "chroma_db", FixedEmbeddings())
    index.index_file(source)
    index.collection.add(
        ids=["stale"],
        documents=["obsolete"],
        embeddings=[FixedEmbeddings().embed_query("obsolete")],
        metadatas=[{"source": str(source.resolve()), "content_hash": "old", "filename": source.name, "ordinal": 9}],
    )
    assert index.count() == 2
    assert index.index_file(source) == "indexed"
    assert index.collection.get(include=["documents"])["documents"] == ["Alpha evidence"]


def test_invalid_path_fails_before_loading_embedding_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from open_rag import index as index_module

    def unexpected_model_load(**kwargs):
        raise AssertionError("embedding model loaded before path validation")

    monkeypatch.setattr(index_module, "HuggingFaceEmbeddings", unexpected_model_load)
    index = index_module.DocumentIndex(tmp_path / "chroma_db")
    with pytest.raises(DocumentError, match="missing.txt"):
        index.index_file(tmp_path / "missing.txt")


def test_searches_all_documents_and_gates_unrelated_results(tmp_path: Path) -> None:
    from open_rag.index import DocumentIndex

    index = DocumentIndex(tmp_path / "chroma_db", FixedEmbeddings())
    for name, text in (("alpha.txt", "Alpha evidence"), ("beta.txt", "Beta evidence")):
        source = tmp_path / name
        source.write_text(text, encoding="utf-8")
        index.index_file(source)

    related = index.search("alpha question")
    assert [hit.filename for hit in related.hits] == ["alpha.txt", "beta.txt"]
    assert [hit.similarity for hit in related.hits] == [1.0, 0.0]
    assert [hit.filename for hit in related.usable] == ["alpha.txt"]
    assert related.embed_ms >= 0 and related.query_ms >= 0

    unrelated = index.search("unrelated question")
    assert unrelated.hits and unrelated.usable == []


def test_search_considers_more_candidates_than_it_supplies(tmp_path: Path) -> None:
    from open_rag.index import DocumentIndex

    index = DocumentIndex(tmp_path / "chroma_db", FixedEmbeddings())
    for number in range(13):
        source = tmp_path / f"alpha{number}.txt"
        source.write_text(f"Alpha evidence {number}", encoding="utf-8")
        index.index_file(source)
    outcome = index.search("alpha question")
    assert len(outcome.hits) == 12
    assert len(outcome.usable) == 3
    assert outcome.decisions.count("selected") == 3
    assert len(outcome.decisions) == len(outcome.hits)


def test_reingest_upgrades_legacy_chunks_even_when_source_is_unchanged(tmp_path: Path) -> None:
    from open_rag.documents import load_document
    from open_rag.index import DocumentIndex

    source = tmp_path / "notes.txt"
    source.write_text("Alpha evidence", encoding="utf-8")
    index_path = tmp_path / "chroma_db"
    index = DocumentIndex(index_path, FixedEmbeddings())
    index.collection.add(
        ids=["legacy-id"],
        documents=["Alpha evidence"],
        embeddings=[FixedEmbeddings().embed_query("Alpha evidence")],
        metadatas=[{
            "source": str(source.resolve()),
            "filename": source.name,
            "content_hash": load_document(source).digest,
            "ordinal": 0,
        }],
    )

    assert index.index_file(source) == "indexed"
    stored = index.collection.get(include=["metadatas"])
    assert len(stored["ids"]) == 1
    assert stored["ids"][0] != "legacy-id"
    assert stored["metadatas"][0]["chunk_format"] == 2
    assert index.index_file(source) == "unchanged"
    reopened = DocumentIndex(index_path, FixedEmbeddings())
    assert reopened.collection.get(include=["metadatas"])["metadatas"][0]["chunk_format"] == 2


def test_failed_upgrade_keeps_legacy_chunks(tmp_path: Path) -> None:
    from open_rag.index import DocumentIndex

    source = tmp_path / "notes.txt"
    source.write_text("Alpha evidence", encoding="utf-8")
    index = DocumentIndex(tmp_path / "chroma_db", FixedEmbeddings())
    index.collection.add(
        ids=["legacy-id"],
        documents=["Alpha evidence"],
        embeddings=[FixedEmbeddings().embed_query("Alpha evidence")],
        metadatas=[{"source": str(source.resolve()), "filename": source.name, "content_hash": "old", "ordinal": 0}],
    )

    class FailingEmbeddings:
        def embed_documents(self, texts):
            raise RuntimeError("embedding failed")

    index._embeddings = FailingEmbeddings()
    with pytest.raises(RuntimeError, match="embedding failed"):
        index.index_file(source)
    assert index.collection.get()["ids"] == ["legacy-id"]


def test_selected_context_respects_budget_without_truncating_passages(tmp_path: Path) -> None:
    from open_rag.index import DocumentIndex
    from open_rag.qa import build_context

    source_dir = tmp_path / ("p" * 200)
    source_dir.mkdir()
    index = DocumentIndex(tmp_path / "chroma_db", FixedEmbeddings())
    for number, letter in enumerate("ABC"):
        source = source_dir / f"evidence-{number}.txt"
        source.write_text("Alpha " + letter * 480, encoding="utf-8")
        index.index_file(source)

    outcome = index.search("alpha question")
    assert len(outcome.usable) < 3
    assert "context budget" in outcome.decisions
    assert len(build_context(outcome.usable)) <= 2500
    assert all(hit.text == Path(hit.source).read_text(encoding="utf-8") for hit in outcome.usable)
