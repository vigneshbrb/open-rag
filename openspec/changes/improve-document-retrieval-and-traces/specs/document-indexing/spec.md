# Spec Delta

## Purpose

Maintains a reusable local index of user-selected documents while keeping related text together and preserving the source of every searchable excerpt.

## ADDED Requirements

### Requirement: Preserve useful text boundaries
The system SHALL form bounded chunks that keep nearby sentences, paragraphs, or log lines together where extractable structure permits. It SHALL keep PDF page attribution and never join text from different pages.

#### Scenario: Loosely structured text
- **WHEN** a TXT document has short lines, blank lines, and a long paragraph
- **THEN** its indexed chunks preserve complete nearby lines or paragraphs where they fit, and split overlong text into bounded chunks

#### Scenario: PDF page boundary
- **WHEN** relevant text occurs at the end of one PDF page and the start of another
- **THEN** each chunk retains its own page number and no chunk spans both pages

### Requirement: Refresh chunks after a chunking change
The system SHALL detect when indexed chunks were produced by an older chunking scheme. Re-ingesting an unchanged source file SHALL replace those chunks with current chunks without leaving duplicate or stale search results.

#### Scenario: Existing index upgrade
- **WHEN** a file with unchanged bytes is re-ingested after the chunking scheme changes
- **THEN** its old chunks are replaced, other files remain searchable, and a repeated ingestion reports unchanged

#### Scenario: Failed upgrade
- **WHEN** extraction or embedding fails while rebuilding chunks
- **THEN** the previously indexed version remains available for searching
