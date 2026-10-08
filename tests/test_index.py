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


def test_search_returns_at_most_three_hits(tmp_path: Path) -> None:
    from open_rag.index import DocumentIndex

    index = DocumentIndex(tmp_path / "chroma_db", FixedEmbeddings())
    for number in range(4):
        source = tmp_path / f"alpha{number}.txt"
        source.write_text(f"Alpha evidence {number}", encoding="utf-8")
        index.index_file(source)
    assert len(index.search("alpha question").hits) == 3
