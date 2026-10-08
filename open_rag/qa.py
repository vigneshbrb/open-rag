"""Grounding prompt and source presentation."""

import sys
from time import perf_counter
from typing import TextIO

from langchain_core.messages import SystemMessage
from langsmith import tracing_context

from .index import DocumentIndex
from .index import Hit


REFUSAL = "I cannot find the answer in the document."
PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the provided text context below. "
    "If the context does not contain the answer, say 'I cannot find the answer in the document.' "
    "Do not make up information.Context: {retrieved_chunks}\nQuestion: {user_question}"
)


def _location(hit: Hit) -> str:
    return f"{hit.source}, p. {hit.page}" if hit.page is not None else hit.source


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{number}] {_location(hit)}\n{hit.text}" for number, hit in enumerate(hits, 1))


def build_prompt(question: str, hits: list[Hit]) -> str:
    return PROMPT.format(retrieved_chunks=build_context(hits), user_question=question)


def format_answer(answer: str, hits: list[Hit]) -> str:
    answer = answer.strip()
    if answer == REFUSAL or not hits:
        return answer
    locations = list(dict.fromkeys(_location(hit) for hit in hits))
    return answer + "\n\nSources supplied:\n" + "\n".join(f"- {location}" for location in locations)


def answer_question(
    index: DocumentIndex, question: str, provider: str = "ollama", *, trace: TextIO | None = None, model=None
) -> str:
    trace = trace if trace is not None else sys.stderr
    print(f"[trace] question: {question}", file=trace)
    outcome = index.search(question)
    print(f"[trace] model load: local embedding model elapsed_ms={outcome.load_ms:.1f}", file=trace)
    print(f"[trace] embedding: model=all-MiniLM-L6-v2 dimensions=384 elapsed_ms={outcome.embed_ms:.1f}", file=trace)
    print(f"[trace] Chroma query: collection=documents metric=cosine n_results=3 elapsed_ms={outcome.query_ms:.1f}", file=trace)
    for rank, hit in enumerate(outcome.hits, 1):
        print(
            f"[trace] rank={rank} id={hit.id} cosine_distance={hit.distance:.3f} "
            f"similarity={hit.similarity:.3f} source={_location(hit)}\n{hit.text}",
            file=trace,
        )
    print(f"[trace] retrieval elapsed_ms={outcome.embed_ms + outcome.query_ms:.1f}", file=trace)
    if not outcome.usable:
        print("[trace] generation skipped: no result met similarity threshold 0.3", file=trace)
        print(f"[trace] answer: {REFUSAL}", file=trace)
        return REFUSAL

    context = build_context(outcome.usable)
    prompt = build_prompt(question, outcome.usable)
    print(f"[trace] selected context:\n{context}", file=trace)
    print(f"[trace] system prompt sent to AI:\n{prompt}", file=trace)
    if model is None:
        from .providers import make_model

        model = make_model(provider)
    print(f"[trace] provider={provider} generation starting", file=trace)
    start = perf_counter()
    try:
        with tracing_context(enabled=False):
            response = model.invoke([SystemMessage(content=prompt)])
    except Exception:
        from .providers import ProviderError

        message = (
            "OpenAI request failed. Check credentials and service availability."
            if provider == "openai"
            else "Ollama request failed. Check `ollama serve` and `ollama pull phi4-mini`."
        )
        print(f"[trace] provider={provider} error: {message}", file=trace)
        raise ProviderError(message) from None
    elapsed_ms = (perf_counter() - start) * 1000
    if not isinstance(response.content, str):
        raise ValueError("model returned non-text content")
    answer = response.content.strip()
    print(f"[trace] provider={provider} generation elapsed_ms={elapsed_ms:.1f}", file=trace)
    print(f"[trace] answer: {answer}", file=trace)
    return format_answer(answer, outcome.usable)
