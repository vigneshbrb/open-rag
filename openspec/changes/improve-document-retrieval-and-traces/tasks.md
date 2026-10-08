# Tasks

## 1. Evidence baseline and chunk refresh

- [x] 1.1 Add deterministic TXT and text-based PDF fixtures with paraphrased, scattered, repeated, identifier, unrelated, and instruction-like content; verify tests record the expected passage locations and current retrieval misses before changing search.
- [x] 1.2 Replace fixed-width splitting with bounded, structure-aware chunks and preserved PDF page attribution; verify chunk tests cover paragraphs, lines, overlong text, overlap, whitespace, and page boundaries.
- [x] 1.3 Version chunk IDs and metadata so re-ingesting unchanged files rebuilds legacy chunks safely; verify persistence, failed embedding, stale-version reconciliation, and repeat-ingestion tests.
- [x] 1.4 Document the new chunk rules and required re-ingestion in README; verify the documented ingestion command upgrades a legacy fixture without touching the source file.

## 2. Retrieval and grounding

- [x] 2.1 Query a bounded larger candidate set, gate relevance, and select at most three nonredundant passages within the context budget; verify expected passage selection, candidate exclusion reasons, and source attribution against the fixture set.
- [x] 2.2 Calibrate the similarity gate with positive and unrelated fixture questions; verify paraphrased answers retain expected evidence and unrelated questions refuse without model calls, then record chosen threshold and observed scores.
- [x] 2.3 Update the model prompt to treat retrieved text as untrusted data while retaining the exact refusal and supplied-source behavior; verify prompt-injection, related-but-unanswered, refusal, and source-list tests.
- [x] 2.4 Document candidate selection, refusal limits, and the meaning of `Sources supplied` in README; verify examples match actual CLI output.

## 3. Readable traces

- [x] 3.1 Add one stderr trace formatter with labeled stages, ranked candidate decisions, timings, delimited context and prompt, and clear provider outcome; verify answer and refusal traces against representative fixtures.
- [x] 3.2 Keep redirected traces free of control sequences and credentials while preserving stdout for answers and sources; verify pipe/StringIO, provider-error, and secret-redaction tests.
- [x] 3.3 Update README trace examples and disclosure note; verify the examples against a captured CLI run.

## 4. Integration proof

- [x] 4.1 Run the full test suite and a local end-to-end ingest/ask check across restart and changed-file refresh; verify relevant evidence, refusal, source list, and trace stages.
- [x] 4.2 Measure warm embedding plus retrieval latency over repeated queries with the wider candidate set; record median and range against the existing 200 ms target without including startup or generation.

## Workflow follow-up

- Archive `local-document-qa` first when archive is requested. Then reconcile this change's delta operations with the newly durable capability specs before archiving this change.
