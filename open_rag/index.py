"""Persistent, multi-document Chroma index."""

from hashlib import sha256
from pathlib import Path
from dataclasses import dataclass
import re
from time import perf_counter

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings

from .chunking import Chunk, split_document
from .documents import load_document


CHUNK_FORMAT = 2


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    source: str
    filename: str
    page: int | None
    distance: float
    similarity: float


@dataclass(frozen=True)
class SearchOutcome:
    hits: list[Hit]
    usable: list[Hit]
    decisions: list[str]
    load_ms: float
    embed_ms: float
    query_ms: float


class DocumentIndex:
    def __init__(self, path: str | Path = "./chroma_db", embeddings=None) -> None:
        self.path = Path(path).expanduser().resolve()
        self._embeddings = embeddings
        client = chromadb.PersistentClient(path=str(self.path), settings=Settings(anonymized_telemetry=False))
        self.collection = client.get_or_create_collection("documents", metadata={"hnsw:space": "cosine"})

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"local_files_only": True},
            )
        return self._embeddings

    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def _chunk_id(chunk: Chunk) -> str:
        source_id = sha256(chunk.source.encode("utf-8")).hexdigest()
        return f"{source_id}:{chunk.digest}:v{CHUNK_FORMAT}:{chunk.ordinal}"

    def index_file(self, path: str | Path) -> str:
        document = load_document(path)
        chunks = split_document(document)
        expected = {self._chunk_id(chunk): chunk for chunk in chunks}
        existing = set(self.collection.get(where={"source": document.source})["ids"])
        if existing == set(expected):
            return "unchanged"

        missing = [(id_, chunk) for id_, chunk in expected.items() if id_ not in existing]
        if missing:
            vectors = self.embeddings.embed_documents([chunk.text for _, chunk in missing])
            if len(vectors) != len(missing) or any(len(vector) != 384 for vector in vectors):
                raise ValueError("embedding model must return one 384-dimensional vector per chunk")
            metadata = []
            for _, chunk in missing:
                item = {
                    "source": chunk.source,
                    "filename": chunk.filename,
                    "content_hash": chunk.digest,
                    "chunk_format": CHUNK_FORMAT,
                    "ordinal": chunk.ordinal,
                }
                if chunk.page is not None:
                    item["page"] = chunk.page
                metadata.append(item)
            self.collection.add(
                ids=[id_ for id_, _ in missing],
                documents=[chunk.text for _, chunk in missing],
                embeddings=vectors,
                metadatas=metadata,
            )

        stale = list(existing - set(expected))
        if stale:
            self.collection.delete(ids=stale)
        return "indexed"

    def search(self, question: str, minimum_similarity: float = 0.2) -> SearchOutcome:
        count = self.count()
        if count == 0:
            raise ValueError("No documents indexed. Run `python -m open_rag ingest <paths...>` first.")
        load_start = perf_counter()
        embeddings = self.embeddings
        start = perf_counter()
        vector = embeddings.embed_query(question)
        if len(vector) != 384:
            raise ValueError("query embedding must have 384 dimensions")
        embedded = perf_counter()
        result = self.collection.query(
            query_embeddings=[vector], n_results=min(12, count), include=["documents", "metadatas", "distances"]
        )
        queried = perf_counter()
        hits = []
        for id_, text, metadata, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            hits.append(
                Hit(id_, text, metadata["source"], metadata["filename"], metadata.get("page"), float(distance), similarity)
            )
        selected: list[Hit] = []
        decisions: list[str] = []
        selected_tokens: list[set[str]] = []
        context_chars = 0
        for hit in hits:
            tokens = set(re.findall(r"\w+", hit.text.casefold()))
            cost = len(hit.text) + len(hit.source) + 40
            if hit.similarity < minimum_similarity:
                decision = "below threshold"
            elif any(
                tokens and prior and len(tokens & prior) / len(tokens | prior) >= 0.8
                for prior in selected_tokens
            ):
                decision = "redundant"
            elif len(selected) == 3:
                decision = "selection limit"
            elif context_chars + cost > 2500:
                decision = "context budget"
            else:
                decision = "selected"
                selected.append(hit)
                selected_tokens.append(tokens)
                context_chars += cost
            decisions.append(decision)
        return SearchOutcome(
            hits,
            selected,
            decisions,
            (start - load_start) * 1000,
            (embedded - start) * 1000,
            (queried - embedded) * 1000,
        )
