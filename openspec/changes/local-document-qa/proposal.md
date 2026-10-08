# Proposal

## Why

Users need a local command-line way to ask grounded questions across several PDF and TXT documents without rebuilding an index on every run. They also need to see how each answer was produced, including retrieval and the prompt sent to the model. The repository currently has OpenSpec configuration but no application or specifications.

## What Changes

- Add ingestion for readable TXT and text-based PDF files, with 500-character chunks and 50-character overlap.
- Persist embeddings from `sentence-transformers/all-MiniLM-L6-v2` in one local Chroma collection; update changed documents without duplicate chunks.
- Add interactive and single-question CLI modes that retrieve the top three chunks across all indexed documents, generate a grounded answer, and print a source list.
- Show a trace for every question: query processing, vector search and its ranked results, selected context, populated model prompt, provider call, and answer or refusal. Never print credentials.
- Support Ollama `phi4-mini` by default and configurable OpenAI `gpt-4o-mini` in the first version. Keep provider selection contained so additional providers can be added later.
- Reject invalid or unreadable input and return the required refusal when no usable evidence is found.

## Capabilities

### New Capabilities

- `document-indexing`: Extract, chunk, embed, persist, and refresh multiple local documents.
- `document-qa`: Retrieve indexed evidence, trace each request, answer via a selected model, and report sources through the CLI.

### Modified Capabilities

None.

## Impact

- New Python 3.10+ CLI application and tests.
- Runtime dependencies for text-based PDF extraction, LangChain orchestration, ChromaDB, local sentence-transformer embeddings, Ollama, and OpenAI.
- Local `./chroma_db` data directory; questions and retrieved excerpts leave the machine when OpenAI is selected.
- Trace output includes document excerpts and the final model prompt on the user's terminal.
