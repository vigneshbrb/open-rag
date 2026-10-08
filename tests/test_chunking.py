from open_rag.documents import LoadedDocument, Segment


def test_fixed_chunks_have_exact_overlap_and_page_boundaries() -> None:
    from open_rag.chunking import split_document

    document = LoadedDocument("/tmp/report.pdf", "digest", (Segment("A" * 1000, 1), Segment("B" * 30, 2)))
    chunks = split_document(document)

    assert [len(chunk.text) for chunk in chunks] == [500, 500, 100, 30]
    assert [chunk.page for chunk in chunks] == [1, 1, 1, 2]
    assert [chunk.ordinal for chunk in chunks] == [0, 1, 2, 3]
    assert chunks[0].text[-50:] == chunks[1].text[:50]
    assert chunks[1].text[-50:] == chunks[2].text[:50]


def test_ignores_whitespace_only_chunks() -> None:
    from open_rag.chunking import split_document

    document = LoadedDocument("/tmp/notes.txt", "digest", (Segment(" " * 600, None), Segment("B" * 20, None)))
    assert [chunk.text for chunk in split_document(document)] == ["B" * 20]


def test_separate_paragraphs_remain_separate_chunks() -> None:
    from open_rag.chunking import split_document

    document = LoadedDocument(
        "/tmp/notes.txt", "digest", (Segment("First finding.\n\nSecond finding.", None),)
    )
    assert [chunk.text for chunk in split_document(document)] == ["First finding.", "Second finding."]


def test_log_lines_are_not_cut_midline_when_they_fit() -> None:
    from open_rag.chunking import split_document

    lines = [f"EVENT-{number} " + "x" * 140 for number in range(5)]
    document = LoadedDocument("/tmp/events.txt", "digest", (Segment("\n".join(lines), None),))
    chunks = split_document(document)
    assert all(len(chunk.text) <= 500 for chunk in chunks)
    assert all(any(line in chunk.text for chunk in chunks) for line in lines)
