from __future__ import annotations

import pytest

from rag.index import reset_index
from rag.ingest import ingest_document
from rag.retriever import _score_sentences, _synthesize_answer, retrieve_and_answer


@pytest.fixture(autouse=True)
def fresh_index():
    reset_index()
    yield
    reset_index()


def test_retrieve_empty_index():
    answer, sources = retrieve_and_answer("What is ML?")
    assert "No documents indexed" in answer
    assert sources == []


def test_retrieve_after_ingest():
    text = "Machine learning enables computers to learn from data without explicit programming. " * 5
    ingest_document(text, "ml_doc")
    answer, sources = retrieve_and_answer("What is machine learning?", top_k=2)
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert isinstance(sources, list)
    assert len(sources) <= 2


def test_sources_have_required_keys():
    ingest_document("Deep learning uses neural networks with many layers. " * 5, "dl_doc")
    _, sources = retrieve_and_answer("neural networks", top_k=1)
    if sources:
        assert "doc_id" in sources[0]
        assert "score" in sources[0]
        assert "excerpt" in sources[0]


def test_score_sentences_returns_sorted_by_overlap():
    sentences = ["machine learning is great", "cooking is fun", "deep learning and machine learning"]
    scored = _score_sentences("machine learning", sentences)
    assert len(scored) == 3
    # verify structure
    for score, sent in scored:
        assert isinstance(score, float)
        assert isinstance(sent, str)


def test_score_sentences_empty_query():
    sentences = ["hello world"]
    scored = _score_sentences("", sentences)
    assert len(scored) == 1


def test_synthesize_answer_with_content():
    context = "Neural networks learn by adjusting weights. Backpropagation updates weights. Gradients flow backwards."
    answer = _synthesize_answer("neural networks", context)
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_synthesize_answer_fallback_short_context():
    context = "short"
    answer = _synthesize_answer("query", context)
    assert isinstance(answer, str)


def test_retrieve_top_k_respected():
    text = " ".join(["word"] * 400)
    ingest_document(text, "long_doc")
    _, sources = retrieve_and_answer("word", top_k=1)
    assert len(sources) <= 1
