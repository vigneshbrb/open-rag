# Tasks

## 1. Python CLI foundation

- [x] 1.1 Add Python 3.10+ package metadata, pinned compatible dependencies, and `python -m open_rag` command skeleton; verify `--help` lists `ingest`, `chat`, and `ask`.
- [x] 1.2 Add documented setup for Ollama `phi4-mini`, optional `OPENAI_API_KEY`, and local index location; verify the documented setup and help commands are accurate.

## 2. Document indexing

- [x] 2.1 Implement TXT and text-based PDF extraction with path, readability, and empty-text errors; verify tests cover valid files, scanned/empty PDF, and invalid input without replacing existing data.
- [x] 2.2 Implement 500-character chunks with 50-character overlap, per-page boundaries, metadata, and 384-dimensional local embeddings; verify boundary and attribution tests.
- [x] 2.3 Implement one persistent cosine Chroma collection with canonical-path identity, content-hash reuse, changed-file replacement, and incomplete-update reconciliation; verify restart, repeat, changed-file, and multi-document tests.
- [x] 2.4 Document ingestion and supported file types with runnable examples; verify commands against a small TXT and text-based PDF fixture.

## 3. Retrieval, grounding, and trace

- [x] 3.1 Implement top-three cross-document retrieval, normalized relevance scoring, and a calibrated no-evidence refusal gate; verify relevant and unrelated query tests with recorded scores.
- [x] 3.2 Render the exact system prompt from retrieved context and question, then format a deduplicated `Sources supplied` list; verify verbatim prompt text, source locations, and no source list on refusal.
- [x] 3.3 Emit per-question stderr trace for query embedding, Chroma parameters/results, selected context, full populated prompt, provider stage, timings, and final outcome; verify traces include requested stages and never expose a test API key.
- [x] 3.4 Document trace interpretation, the full-prompt disclosure on the terminal, and what source lists mean; verify sample output matches the CLI.

## 4. Model providers and question CLI

- [x] 4.1 Add Ollama `phi4-mini` as the default and OpenAI `gpt-4o-mini` as a configured option at one model-selection boundary; verify provider selection, local Ollama target, missing key, and provider-error tests.
- [x] 4.2 Implement interactive `chat` and single-question `ask`, keeping answers and sources on stdout and traces on stderr; verify multi-turn use, script output, empty-index error, and clean exit.
- [x] 4.3 Document provider selection, OpenAI data transfer, and command examples; verify examples and error messages against the implemented CLI.

## 5. End-to-end acceptance

- [x] 5.1 Run an end-to-end test on a representative ten-page TXT plus a text-based PDF: verify persistence across restart, changed-file refresh, multiple-document retrieval, source lists, and complete traces.
- [x] 5.2 Run a repeated-ingestion memory check and warm query benchmark; record memory trend and query-embedding plus retrieval duration against the 200 ms target without claiming a universal guarantee.
- [x] 5.3 With local Ollama `phi4-mini` available, verify a document-grounded answer and exact refusal for an unrelated question; record any model-dependent failure rather than treating a stub-model test as proof.
