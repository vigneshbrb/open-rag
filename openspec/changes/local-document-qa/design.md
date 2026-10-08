# Design

## Context

See `proposal.md` for motivation and the two capability specs for behavior. This is a greenfield Python project: there is no application code or existing data format to preserve. The CLI runs on one machine and uses one persistent index at `./chroma_db`.

## Goals / Non-Goals

**Goals:**

- Keep document ownership, index refresh, retrieval, model choice, and trace output easy to inspect.
- Make every answer reproducible enough for a user to see the retrieved passages and exact model prompt.
- Preserve normal stdout for answers in single-question mode; write per-request trace to stderr.

**Non-Goals:**

- OCR for scanned PDFs, remote vector storage, concurrent writers, or automatic filesystem watching.
- GitHub Copilot and Claude integrations in this change. Provider construction stays in one place for later additions, with no plugin framework.
- A guarantee that an LLM never hallucinates; test refusal behavior and guard requests with no usable evidence.

## Decisions

### CLI and orchestration

Use `argparse` with `python -m open_rag ingest <paths...>`, `python -m open_rag chat`, and `python -m open_rag ask <question>`. `chat` and `ask` accept `--provider ollama|openai`, defaulting to Ollama. Use LangChain components for chunking, embeddings, vector search, and model clients; keep the request pipeline explicit so each stage can be traced. A LlamaIndex query engine would hide more of the intermediate request state that the user wants to inspect.

### Extraction, chunking, and index ownership

Read TXT as UTF-8 and use `pypdf` for text-based PDF extraction. Treat each PDF page as a separate segment. Use a fixed-width LangChain text splitter (`chunk_size=500`, `chunk_overlap=50`, character separator) so overlap is exact and chunks never cross PDF pages. Skip whitespace-only chunks. Store source path, display filename, page number, file content hash, and chunk ordinal as metadata; use the canonical source path as document identity. The embedding model is `sentence-transformers/all-MiniLM-L6-v2`, whose vector width is 384.

Use one persistent Chroma collection configured for cosine distance. At ingestion, validate and extract the complete file first. If its content hash matches a complete indexed version with no stale IDs, skip embedding. Otherwise embed the new chunks, add them under versioned IDs, then delete the old IDs. This order protects an existing version from extraction and embedding failures. A failed update after insertion can leave both versions; a later ingestion of that path reconciles them, including when the current file hash matches one version. Other documents remain untouched. An explicit delete command is deferred because no removal workflow was requested.

### Retrieval and refusal

Query the single collection for three results. Trace the collection, cosine metric, `n_results=3`, returned IDs, scores, source locations, and chunk text. Convert cosine distance to a documented similarity score before applying a relevance gate; start at 0.3 and calibrate with known relevant and unrelated questions. If no chunk clears the gate, return the exact refusal without calling a model. If chunks clear it, insert only those chunks into the specified system-prompt template. Keep the template's punctuation verbatim, including `information.Context:`. The model also receives the template's refusal instruction for cases where retrieved text is related but insufficient.

### Source list

Prefix each context chunk with a stable source label and location. After the answer, print a deduplicated list of the chunks supplied to the model, identified by filename and PDF page where present. Label it `Sources supplied` so the CLI does not imply the model relied on every retrieved passage. Refusals have no source list. A separate inline-citation generation pass would change prompt behavior and add a second model call, so the first version uses a transparent post-answer list.

### Model providers and data flow

Construct `ChatOllama(model="phi4-mini")` or `ChatOpenAI(model="gpt-4o-mini")` at one model-selection boundary. `OPENAI_API_KEY` is read from the environment only for OpenAI and is never included in trace events. The Ollama client must target localhost; OpenAI sends the populated prompt and document excerpts to its service. Show the selected provider before the call. Report a clear error when Ollama is unavailable, its model is missing, or OpenAI credentials or service calls fail.

### Trace output and timing

Use one small trace writer in the request path. For each question, print stage names and elapsed times: query embedding, Chroma search with parameters, ranked results, selected context, populated system prompt, provider request, and final answer or refusal. Print the full text sent to the model, as requested, to stderr; never print credentials, headers, or environment values. In single-question mode, stdout contains the answer and source list only. Do not persist trace files by default. Measure the warm query-embedding plus Chroma search interval separately from model generation for the 200 ms target.

## Risks / Trade-offs

- [Retrieved text may still be insufficient despite a good similarity score] → Keep the model refusal instruction and test unrelated questions; report that no score guarantees factuality.
- [Full prompt traces expose document content on the terminal] → Trace only to the invoking user's stderr, do not persist logs by default, and document terminal capture implications.
- [OpenAI mode transmits document excerpts] → Make provider choice explicit and keep Ollama as the default.
- [Chroma updates are not transactional across add and delete] → Add new chunks before deleting old chunks; reconcile leftover versions on the next ingestion.
- [200 ms depends on hardware and warm caches] → Benchmark embedding plus retrieval after warm-up on a representative ten-page file and report measured time; do not include startup or generation.

## Migration Plan

No existing application data requires migration. The initial install creates `./chroma_db` on first ingestion. Rollback removes the application without modifying source documents; the local index can be retained for reuse or deleted by the user.
