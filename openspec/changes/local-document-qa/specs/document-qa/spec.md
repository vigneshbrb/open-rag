# Spec Delta

## Purpose

Lets users ask questions across indexed local documents, inspect the evidence and model request for each question, and receive grounded answers with source references.

## ADDED Requirements

### Requirement: Ask through the CLI
The system SHALL provide interactive question entry and a single-question command suitable for scripts. Questions SHALL search all indexed documents by default and report a clear error when no documents are indexed.

#### Scenario: Interactive session
- **WHEN** a user starts chat with an existing index and enters multiple questions
- **THEN** the system answers each question without requiring reindexing

#### Scenario: Single question
- **WHEN** a user supplies one question on the command line
- **THEN** the system prints one answer and exits

### Requirement: Retrieve relevant evidence
For each question, the system SHALL embed the query and retrieve at most the three most relevant indexed chunks using cosine similarity or L2 distance. Retrieved chunks SHALL retain source metadata and relevance information.

#### Scenario: Multiple documents
- **WHEN** the index contains multiple documents and a user asks a question
- **THEN** the search considers chunks from all indexed documents and returns at most three ranked results

### Requirement: Ground answers and refuse unsupported questions
The system SHALL use only retrieved context to answer. It SHALL return exactly `I cannot find the answer in the document.` when no usable evidence is retrieved and SHALL instruct the model to use that same refusal when the retrieved context does not answer the question.

#### Scenario: No relevant evidence
- **WHEN** an unrelated question yields no usable evidence
- **THEN** the system prints `I cannot find the answer in the document.` without inventing an answer

#### Scenario: Relevant evidence
- **WHEN** retrieved context answers a question
- **THEN** the system sends the question and that context to the selected model and prints the resulting answer

### Requirement: Preserve the grounding prompt
The system SHALL populate and use this exact system-prompt template, including its wording and punctuation, with `{retrieved_chunks}` and `{user_question}` replaced by their values:

```text
You are a helpful assistant. Answer the user's question using ONLY the provided text context below. If the context does not contain the answer, say 'I cannot find the answer in the document.' Do not make up information.Context: {retrieved_chunks}
Question: {user_question}
```

#### Scenario: Model request
- **WHEN** a question has usable retrieved context
- **THEN** the model request contains the populated template as its system prompt

### Requirement: Show sources after the answer
The system SHALL print a source list after an evidence-based answer, identifying the retrieved chunks supplied as context by filename and PDF page where applicable. It SHALL not claim a source was used if it was not supplied to the model.

#### Scenario: Multi-source answer
- **WHEN** context contains chunks from different files or PDF pages
- **THEN** the source list distinguishes those locations and avoids duplicate entries

### Requirement: Trace each question
For every question, the CLI SHALL display processing stages, the vector database query and ranked results with relevance and text, the context selected, the fully populated prompt sent to the model, the selected provider, and the answer or refusal. It SHALL include timings for retrieval and model generation where applicable and SHALL never display API keys or other credentials.

#### Scenario: Answer trace
- **WHEN** a user asks a question with usable evidence
- **THEN** the trace shows query embedding, vector search and its results, selected context, final model prompt, provider call, and output

#### Scenario: Refusal trace
- **WHEN** a question has no usable evidence
- **THEN** the trace shows retrieval and the reason generation was skipped

### Requirement: Select a model provider
The system SHALL support local Ollama `phi4-mini` by default and configurable OpenAI `gpt-4o-mini`. OpenAI credentials SHALL come from a nonprinted environment variable. Ollama mode SHALL keep document content and questions on the local machine.

#### Scenario: Local model
- **WHEN** the user selects Ollama or leaves the provider at its default
- **THEN** the system sends generation requests to the local Ollama service

#### Scenario: OpenAI model
- **WHEN** the user selects OpenAI with valid credentials
- **THEN** the system sends the populated prompt to OpenAI and labels the provider in the trace

### Requirement: Meet retrieval target
After models and the index are loaded, the system SHALL target less than 200 ms for local query embedding and top-three vector retrieval on a representative ten-page text document; generation and startup time are excluded.

#### Scenario: Warm retrieval benchmark
- **WHEN** a representative indexed ten-page text document is queried after warm-up
- **THEN** the measured query-embedding and vector-search duration is reported and compared with the 200 ms target
