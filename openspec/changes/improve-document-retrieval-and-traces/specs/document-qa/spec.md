# Spec Delta

## Purpose

Answers questions from indexed local documents using inspectable evidence, refuses unsupported requests, and explains each retrieval decision in a readable terminal trace.

## ADDED Requirements

### Requirement: Find evidence despite wording differences
The system SHALL consider more search candidates than it supplies to the model, remove redundant passages, and select at most three relevant chunks from all indexed documents. A paraphrased question SHALL be eligible to retrieve evidence without repeating its exact keywords.

#### Scenario: Paraphrased question
- **WHEN** a question describes a fact using different words from an indexed passage
- **THEN** that passage can be selected when its relevance meets the calibrated evidence gate

#### Scenario: Scattered and repeated text
- **WHEN** relevant passages are separated by repetitive, loosely structured text
- **THEN** the selected evidence favors distinct relevant passages over near-duplicate chunks

### Requirement: Keep selected evidence bounded
The system SHALL keep the selected context within a defined size limit suitable for the configured model. It SHALL retain source location for every selected passage and SHALL not silently truncate a passage after selection.

#### Scenario: Excess candidates
- **WHEN** relevant candidates exceed the context limit
- **THEN** the system selects a fitting subset and traces which candidates were excluded

### Requirement: Refuse without sufficient evidence
The system SHALL refuse with exactly `I cannot find the answer in the document.` when no candidate passes the calibrated evidence gate. It SHALL give the model the same refusal instruction when selected text is related but does not answer the question.

#### Scenario: Unrelated question
- **WHEN** indexed documents contain no sufficiently relevant evidence
- **THEN** the system skips generation, prints the exact refusal, and supplies no source list

#### Scenario: Related but unanswered question
- **WHEN** selected passages relate to a question but do not contain its answer
- **THEN** the model is instructed to return the exact refusal instead of filling the gap

### Requirement: Treat document text as untrusted evidence
The system SHALL present retrieved document text as data rather than instructions and SHALL instruct the model to ignore commands found inside it. The displayed source list SHALL contain only passages actually supplied to the model.

#### Scenario: Instruction inside a document
- **WHEN** an indexed passage tells the assistant to ignore its rules or reveal secrets
- **THEN** the request retains the application's grounding and refusal instructions, and the passage cannot change provider choice or access credentials

#### Scenario: Source attribution
- **WHEN** a response uses selected context
- **THEN** its `Sources supplied` list identifies only the supplied document locations

### Requirement: Show a readable question trace
The system SHALL write a structured per-question trace to stderr with labeled stages, ranked candidates, relevance and selection decisions, source locations, timings, selected context, the full populated prompt, provider, and final answer or refusal. It SHALL never print credentials.

#### Scenario: Interactive terminal
- **WHEN** a user asks a question in an interactive terminal
- **THEN** the trace separates search, evidence selection, model request, and outcome into readable sections

#### Scenario: Redirected trace
- **WHEN** stderr is redirected to a file or pipe
- **THEN** the trace remains readable plain text without terminal control sequences, while stdout remains answer and sources only

#### Scenario: No evidence
- **WHEN** the system refuses before generation
- **THEN** the trace states why generation was skipped and does not claim a model call occurred
