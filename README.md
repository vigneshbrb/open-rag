# Local document Q&A

Python 3.10+ CLI for text-based PDF and UTF-8 TXT documents. The index is stored in `./chroma_db`, relative to the directory where commands run. Run commands from the same directory to reuse it.

## Setup

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m open_rag --help
```

For local answers, install [Ollama](https://ollama.com/), start its local service, and pull the model:

```sh
ollama serve
ollama pull phi4-mini
```

Run `ollama serve` in another terminal if the service is not already running. Ollama is the default answer provider.

Download the embedding model once during setup, before offline use:

```sh
.venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("sentence-transformers/all-MiniLM-L6-v2")'
```

OpenAI is optional. Set `OPENAI_API_KEY` in the environment before selecting `--provider openai`. OpenAI mode sends the question and retrieved document excerpts to OpenAI.

See `python -m open_rag ingest --help`, `chat --help`, and `ask --help` for command options.

## Index documents

```sh
.venv/bin/python -m open_rag ingest notes.txt report.pdf
```

Supply one or more UTF-8 `.txt` files or text-based `.pdf` files. Scanned PDFs need OCR and are not supported. Chunks stay within 500 characters and one PDF page. Paragraph and line boundaries are kept where possible; long passages can overlap by up to 50 characters. Re-run `ingest` for files indexed by an older version to rebuild their chunks, even when the files have not changed. The first such run reports `indexed`; later unchanged runs report `unchanged`. Other indexed files remain searchable, and ingestion never edits the source file. An invalid file prints an error and does not replace its previous indexed version. Runtime embedding loads only local model files. If the setup download was skipped, ingestion reports that the model is missing.

## Understand each answer

Every question prints a trace to stderr. Stages show embedding and search times, up to 12 ranked candidates with selection decisions, selected context, the full system and user messages sent to the model, and the answer or refusal. For example, an unrelated question produces this abbreviated trace:

```text
[Question]
  Who wrote Hamlet?
[Retrieval]
  model load: all-MiniLM-L6-v2 elapsed_ms=...
  embedding: dimensions=384 elapsed_ms=...
  Chroma query: collection=documents metric=cosine candidates=1 elapsed_ms=...
  rank=1 similarity=0.008 cosine_distance=0.992 decision=below threshold source=/path/notes.txt id=...
  text:
    | Alpha evidence
  retrieval elapsed_ms=...
[Evidence]
  selected=0 threshold=0.2 context_budget=2500
  selection: no usable evidence
[Model]
  skipped: no selected evidence
[Outcome]
  refusal: I cannot find the answer in the document.
```

For an answered question, `[Evidence]` includes `selected context:` and `[Model]` includes `full prompt:` with separate `System:` and `User:` blocks, provider, and generation time. The prompt trace contains the question and document excerpts. Terminal recording or stderr redirection can retain them. The OpenAI API key is redacted from the trace, and control characters in document text are escaped. The embedding library can print its own model-loading progress before `[Retrieval]` on a cold run. The answer goes to stdout, followed by `Sources supplied`: a deduplicated list of selected passages sent to the model. That list identifies available evidence; it does not claim the model used every passage for every sentence. A refusal has no source list.

## Ask questions

Run from the directory containing the index:

```sh
.venv/bin/python -m open_rag ask 'What does the report say?'
.venv/bin/python -m open_rag chat
```

`chat` accepts questions until `exit`, `quit`, or end of input. Both commands search every indexed document. Ollama `phi4-mini` is the default and uses the local service at `127.0.0.1:11434`.

Search examines up to 12 vector candidates, then supplies at most three distinct passages within a 2,500-character context budget. The current cosine-similarity gate is 0.2. If no candidate passes it, the CLI prints exactly `I cannot find the answer in the document.` without calling a model. A related passage can still lack the answer; the model is instructed to give the same refusal. The score is a retrieval filter, not proof that an answer is true. The prompt treats document text as untrusted evidence and tells the model to ignore commands inside it.

To use OpenAI `gpt-4o-mini`, set `OPENAI_API_KEY` in your shell environment and select it explicitly:

```sh
.venv/bin/python -m open_rag ask 'What does the report say?' --provider openai
.venv/bin/python -m open_rag chat --provider openai
```

OpenAI receives the question and selected document excerpts in the prompt. Missing credentials produce `OPENAI_API_KEY is required for --provider openai`. If Ollama is unavailable, the CLI asks you to check `ollama serve` and `ollama pull phi4-mini`. An empty index reports `No documents indexed` and prompts you to run `ingest`.
