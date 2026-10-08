# Verification snapshot

Run on macOS arm64, Python 3.12.14, 2026-10-08. The ten-page-sized UTF-8 fixture repeated 2,233 characters of text ten times and produced 50 chunks. The local `all-MiniLM-L6-v2` model and Chroma collection were warm before timing.

- 30 unchanged re-ingestions: all reported `unchanged`; traced Python heap retained 32 additional bytes after garbage collection.
- Process peak RSS increased 376,832 bytes across those re-ingestions. Peak RSS cannot decrease, so this is a bounded-run observation, not proof of no leak.
- 12 warm queries, embedding plus top-three Chroma search: median 12.0 ms, range 11.7–13.1 ms; below the 200 ms target on this machine. Model loading and answer generation excluded.
- Relevance spot check on a TXT and text-based PDF: `Where is Alpha evidence?` scored 0.819 and 0.608; `Who wrote Hamlet?` scored 0.121 and 0.000. The 0.3 gate accepted the former and refused the latter. These scores do not guarantee behavior on every document.

Automated tests cover index persistence, refresh, multi-document retrieval, prompt text, source list, refusal gate, traces, provider selection, and CLI output. Live Ollama behavior is checked separately when its service and `phi4-mini` model are available.

Live `phi4-mini:latest` check on the TXT and PDF fixture:

- `Where is Alpha evidence?` returned `Answer: In the text file at [1].` with `notes.txt` and `report.pdf` in `Sources supplied` (both were provided as context).
- `What year was Alpha evidence written?` retrieved related context but the model returned exactly `I cannot find the answer in the document.`; no source list was printed.
- `Who wrote Hamlet?` was refused with the same exact text before generation because both retrieved similarities were below 0.3.
