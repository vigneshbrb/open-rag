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

Supply one or more UTF-8 `.txt` files or text-based `.pdf` files. Scanned PDFs need OCR and are not supported. Indexing the same unchanged file again reports `unchanged`; editing it replaces its old chunks. Other indexed files remain searchable. An invalid file prints an error and does not replace its previous indexed version. Runtime embedding loads only local model files. If the setup download was skipped, ingestion reports that the model is missing.

## Understand each answer

Every question prints a trace to stderr. The trace shows query embedding time, the Chroma query (`documents`, cosine distance, top 3), each ranked chunk with distance and similarity, the selected context, the full populated system prompt, the provider call, and the answer or refusal. For example:

```text
[trace] question: What does the report say?
[trace] model load: local embedding model elapsed_ms=...
[trace] embedding: model=all-MiniLM-L6-v2 dimensions=384 elapsed_ms=...
[trace] Chroma query: collection=documents metric=cosine n_results=3 elapsed_ms=...
[trace] rank=1 id=... cosine_distance=... similarity=... source=/path/report.pdf, p. 2
...
[trace] selected context:
...
[trace] system prompt sent to AI:
...
[trace] provider=ollama generation starting
[trace] answer: ...
```

The prompt trace contains the question and document excerpts. Terminal recording or stderr redirection can retain them. API keys are not printed. The answer goes to stdout, followed by `Sources supplied`: a deduplicated list of retrieved passages sent to the model. That list identifies available evidence; it does not claim the model used every passage for every sentence. A refusal has no source list.

## Ask questions

Run from the directory containing the index:

```sh
.venv/bin/python -m open_rag ask 'What does the report say?'
.venv/bin/python -m open_rag chat
```

`chat` accepts questions until `exit`, `quit`, or end of input. Both commands search every indexed document. Ollama `phi4-mini` is the default and uses the local service at `127.0.0.1:11434`.

To use OpenAI `gpt-4o-mini`, set `OPENAI_API_KEY` in your shell environment and select it explicitly:

```sh
.venv/bin/python -m open_rag ask 'What does the report say?' --provider openai
.venv/bin/python -m open_rag chat --provider openai
```

OpenAI receives the question and selected document excerpts in the prompt. Missing credentials produce `OPENAI_API_KEY is required for --provider openai`. If Ollama is unavailable, the CLI asks you to check `ollama serve` and `ollama pull phi4-mini`. An empty index reports `No documents indexed` and prompts you to run `ingest`.
