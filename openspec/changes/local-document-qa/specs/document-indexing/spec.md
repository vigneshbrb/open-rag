# Spec Delta

## Purpose

Maintains a reusable local search index for multiple user-selected text documents while preserving the source of every retrieved excerpt.

## ADDED Requirements

### Requirement: Accept local documents
The system SHALL accept paths to readable `.txt` and text-based `.pdf` files and extract their text. It SHALL report a clear error for invalid paths, unreadable files, unsupported formats, or files with no extractable text.

#### Scenario: Text and PDF ingestion
- **WHEN** a user indexes a readable TXT file and a text-based PDF
- **THEN** the system extracts text from both and retains the source filename and PDF page number where applicable

#### Scenario: Invalid or unreadable document
- **WHEN** a supplied path is missing, unreadable, unsupported, or yields no text
- **THEN** the system identifies that file and the reason it cannot be indexed without replacing any previously indexed version of it

### Requirement: Chunk and embed extracted text
The system SHALL split extracted text into chunks of at most 500 characters with 50-character overlap between successive chunks from the same text segment. It SHALL embed nonempty chunks with `sentence-transformers/all-MiniLM-L6-v2` into 384-dimensional vectors.

#### Scenario: Chunk boundaries
- **WHEN** an extractable segment exceeds 500 characters
- **THEN** successive chunks contain at most 500 characters and overlap by 50 characters

#### Scenario: Page attribution
- **WHEN** a PDF contains text on multiple pages
- **THEN** each chunk retains its originating page number and no chunk spans two pages

### Requirement: Persist and refresh a multi-document index
The system SHALL persist all indexed documents in `./chroma_db`, preserve them across process restarts, and avoid duplicate chunks when the same file is indexed again. A changed file SHALL replace its earlier indexed chunks while other documents remain available.

#### Scenario: Restart and reuse
- **WHEN** the application restarts after indexing documents
- **THEN** the existing index can answer queries without re-embedding unchanged files

#### Scenario: Changed file
- **WHEN** a previously indexed file is edited and indexed again
- **THEN** searches use its new chunks and exclude its old chunks while retaining other indexed files

#### Scenario: Repeated unchanged file
- **WHEN** an unchanged file is indexed again
- **THEN** the index contains no additional copies of its chunks
