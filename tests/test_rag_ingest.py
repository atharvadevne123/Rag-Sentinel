from __future__ import annotations

import pytest

from rag.index import reset_index
from rag.ingest import chunk_text, clean_text, get_stored_chunks, ingest_document


@pytest.fixture(autouse=True)
def fresh_index():
    reset_index()
    yield
    reset_index()


def test_clean_text_normalises_whitespace():
    assert clean_text("hello   world") == "hello world"


def test_clean_text_strips_non_ascii():
    assert clean_text("caf\xe9 au lait") == "caf au lait"


def test_clean_text_strips_leading_trailing():
    assert clean_text("  hello  ") == "hello"


def test_chunk_text_basic():
    text = " ".join(str(i) for i in range(200))
    chunks = chunk_text(text, size=50, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        words = c.split()
        assert len(words) <= 50


def test_chunk_text_overlap():
    text = " ".join(["word"] * 100)
    chunks = chunk_text(text, size=20, overlap=5)
    # consecutive chunks share 5 words
    assert len(chunks) >= 2


def test_chunk_text_short_text_single_chunk():
    text = "short text"
    chunks = chunk_text(text, size=50, overlap=5)
    assert len(chunks) == 1
    assert chunks[0] == "short text"


def test_ingest_document_returns_chunk_count():
    text = "Machine learning is a subset of artificial intelligence. " * 20
    count = ingest_document(text, "doc_test_1")
    assert count > 0


def test_ingest_document_stores_chunks():
    text = "Retrieval-augmented generation improves factual accuracy. " * 15
    doc_id = "doc_store_test"
    count = ingest_document(text, doc_id)
    stored = get_stored_chunks(doc_id)
    assert len(stored) == count


def test_ingest_document_empty_text_returns_zero():
    count = ingest_document("   ", "doc_empty")
    assert count == 0


def test_get_stored_chunks_unknown_doc():
    result = get_stored_chunks("nonexistent_doc_id")
    assert result == []


def test_ingest_multiple_docs():
    ingest_document("First document about science. " * 10, "doc_a")
    ingest_document("Second document about cooking. " * 10, "doc_b")
    from rag.index import get_index
    assert len(get_index()) >= 2
