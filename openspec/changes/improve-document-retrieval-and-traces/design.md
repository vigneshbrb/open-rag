# Design

## Context

See `proposal.md` for motivation and the two delta specs for behavior. At the baseline, `open_rag/chunking.py` made fixed 500-character chunks with 50-character overlap. `DocumentIndex.search()` asked Chroma for three cosine hits and filtered them at similarity 0.3. `answer_question()` wrote a detailed but flat stderr trace, built one exact system prompt, and returned the model's text with `Sources supplied`. The initial `local-document-qa` change is complete but not archived, so its capability specs are not yet in `openspec/specs/`.

## Goals / Non-Goals

**Goals:**

- Improve evidence recall and readability without changing the default local provider, source-document ownership, or stdout answer contract.
- Keep candidate ranking, selection, refusal, and trace output inspectable in one request path.
- Preserve safe index refresh across a change to chunk boundaries.

**Non-Goals:**

- OCR, a second search service, a new embedding model, or a claim that prompt wording guarantees factual answers.
- Arbitrary ingestion of executable document instructions, automatic file watching, or concurrent index writers.

## Decisions

### Structure-aware chunks and index compatibility

Keep a 500-character maximum and PDF page boundary. Prefer paragraph breaks, then line breaks, sentences, spaces, and finally characters for overlong text. Retain up to 50 characters of overlap for split long passages; do not force overlap across independent paragraphs or pages. This is a small change within the current splitter and preserves source metadata. A specialized parser for every file format adds complexity without evidence that this corpus needs it.

Add a chunk-format version to stored metadata and chunk IDs. Treat missing version as legacy. `index_file()` compares expected IDs against all IDs for the source; unchanged source bytes with legacy IDs therefore rebuild. Keep the current add-new-before-delete-old order and reconcile stale IDs on a later ingestion. If extraction or embedding fails before insertion, the old version remains. Document that previously indexed files use old chunks until each is ingested again; do not silently rewrite the entire index at query time.

### Candidate search and evidence selection

Keep the existing local embedding model and cosine Chroma collection. Query up to 12 candidates (or the collection size), apply the evidence gate to each, then select at most three nonredundant passages within a 2,500-character context budget including source labels. Prefer higher similarity; skip passages with substantial text overlap with already selected passages. Preserve their original score, rank, and exclusion reason for tracing. The final context stays small enough for the default local model. Increasing only the top-three query or lowering the 0.3 gate would improve some recall but also bring more irrelevant or repeated text into the prompt.

Calibration on the fixed evaluation set lowered the gate from 0.3 to 0.2: a paraphrased PDF hit scored 0.216, while an unrelated question peaked at 0.056. The identifier example scored 0.736. Record expected source locations, candidate rank, selected context, and refusal outcome. These scores are fixture-specific; real user files remain necessary for broader calibration. If exact identifiers become a demonstrated miss, a later change can add lexical retrieval; a second index is not required by the present evidence.

### Grounding and untrusted documents

Keep deterministic pre-generation refusal when no passage clears the gate. Replace the initial change's exact prompt wording with an instruction that marks retrieved passages as untrusted data, forbids following instructions inside them, requires answers from supplied text only, and gives the existing exact refusal for absent answers. Maintain source labels in context and show `Sources supplied` only for selected passages. Avoid claiming that source presence proves each sentence: this change does not add a separate factuality model or unverifiable citation checker. Test against document text that attempts to override the assistant's instructions and against related text lacking the answer.

### Readable traces

Move trace presentation to one small formatter used by `answer_question()`. Write stage headings to stderr in request order: Question, Retrieval, Evidence, Model, Outcome. Indent each ranked candidate with rank, score, source, and `selected` or an exclusion reason. Put context and the full populated prompt in clearly delimited multiline blocks. Keep timings beside their stage and state when generation was skipped. Use plain text always; optional ANSI emphasis may be used only when stderr is a TTY. Redirected stderr and injected `StringIO` traces contain no control codes. Never print credentials or provider exception text that could include them. The answer and source list stay on stdout.

## Risks / Trade-offs

- [A broader candidate query increases retrieval time] -> Measure warm query embedding plus search against the existing 200 ms target and record any regression.
- [A 0.2 cosine score is not a universal relevance guarantee] -> Keep positive and negative evaluation examples, refusal, and traced decisions.
- [Boundary-aware chunks still lose structure in poorly extracted PDFs] -> Fall back to bounded character splits and retain page attribution; do not promise OCR.
- [An interrupted Chroma add/delete can expose old and new chunks] -> Keep versioned IDs and reconcile on the next ingestion, matching the current failure model.
- [Full prompt traces reveal document excerpts on a terminal] -> Preserve the explicit existing disclosure, write only to the invoking user's stderr, and document capture risks.
- [The prior change's exact prompt and chunk requirements conflict with this design] -> Keep this new change on the same capability paths and reconcile delta operations against the durable specs before archiving this change.

## Migration Plan

Implement and verify the new chunk format before changing retrieval. Users re-run `ingest` for indexed files to rebuild them; unchanged source files are not edited. Keep old chunks searchable until each replacement succeeds. Rollback uses the previous application version with a fresh index built by re-ingesting source files, since mixed chunk formats can be read but old code cannot infer new chunk boundaries. Archive `local-document-qa` before this change when archiving is requested, then reconcile this change's deltas with the durable `document-indexing` and `document-qa` specs.
