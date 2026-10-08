# Proposal

## Why

Questions over loosely structured TXT and PDF content can miss relevant evidence because ingestion cuts text every 500 characters and search considers only the first three vector hits. The per-question trace contains the needed facts but is hard to scan as an unstructured stream of text.

## What Changes

- Preserve paragraph, line, and PDF page boundaries where possible while keeping chunks bounded; retain deterministic source attribution and safe index refresh.
- Evaluate a broader set of semantic search candidates, then select a small, nonredundant evidence set within a model context budget. Calibrate the refusal gate against relevant and unrelated examples rather than relying on the initial fixture alone.
- Guard answers against unsupported claims and instructions embedded in retrieved documents. Keep the exact refusal for missing evidence and show which passages were supplied.
- Present each question's stderr trace as labeled stages with readable ranked results, decisions, timings, selected context, full model prompt, and outcome. Preserve plain, redirectable output and credential redaction.
- Add representative evaluation fixtures for paraphrases, scattered facts, identifiers, unrelated questions, and prompt-like text inside documents.

## Capabilities

### New Capabilities

- `document-indexing`: Extend the pending document-indexing behavior with structure-aware, bounded chunks and safe reindexing.
- `document-qa`: Extend the pending document-qa behavior with candidate selection, grounding guards, and readable traces.

### Modified Capabilities

None. The project has no durable `openspec/specs/` entries yet. The completed `local-document-qa` change still holds the initial versions of these same capability paths; its fixed-width chunking, three-result search, and exact-prompt requirements need reconciliation before both changes are archived.

## Impact

- Planning affects `open_rag/chunking.py`, `open_rag/index.py`, `open_rag/qa.py`, related tests, and trace documentation.
- Reindexing must replace chunks produced by the old chunking scheme even when source bytes have not changed. Existing source documents remain untouched.
- `ask` stdout remains the answer and source list; traces remain on stderr. No new service or model dependency is planned.
