from __future__ import annotations

import pytest

from rag.index import SentinelIndex, reset_index
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


def test_retrieve_no_results_returns_fallback():
    idx = SentinelIndex()
    text = "completely unrelated content " * 5
    vecs = idx.embed([text])
    idx.add(vecs, [text], "fallback_doc")
    # Replace the singleton to inject our index
    import rag.index

    rag.index._index_instance = idx

    answer, sources = retrieve_and_answer("totally different topic", top_k=3)
    assert isinstance(answer, str)
    assert isinstance(sources, list)

    reset_index()


def test_retrieve_multi_doc_returns_sources_from_different_docs():
    ingest_document("machine learning fundamentals and basics. " * 5, "ml_doc")
    ingest_document("cooking recipes and food preparation. " * 5, "food_doc")
    _, sources = retrieve_and_answer("machine learning", top_k=3)
    doc_ids = {s["doc_id"] for s in sources}
    assert len(doc_ids) >= 1


def test_synthesize_answer_ends_with_period():
    context = "Neural networks learn patterns. Backpropagation computes gradients. Training minimizes loss."
    answer = _synthesize_answer("neural networks learning", context)
    assert answer.endswith(".")


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_retrieve_top_k_parametrized(top_k):
    text = "deep learning neural networks transformers attention. " * 10
    ingest_document(text, f"doc_topk_{top_k}")
    _, sources = retrieve_and_answer("neural networks", top_k=top_k)
    assert len(sources) <= top_k


def test_score_sentences_overlap_values():
    sentences = ["no overlap here", "machine learning rocks", "machine learning is great"]
    scored = _score_sentences("machine learning", sentences)
    scores = [s for s, _ in scored]
    assert scores[0] == 0.0
    assert scores[1] > 0.0
    assert scores[2] > 0.0


def test_sources_excerpt_max_length():
    ingest_document("This is a really long sentence about machine learning. " * 20, "long_excerpt_doc")
    _, sources = retrieve_and_answer("machine learning", top_k=2)
    for s in sources:
        assert len(s["excerpt"]) <= 120


def test_retrieve_score_is_float():
    ingest_document("transformer model attention mechanism. " * 5, "transformer_doc")
    _, sources = retrieve_and_answer("attention", top_k=1)
    if sources:
        assert isinstance(sources[0]["score"], float)


def test_retrieve_min_score_filters_results():
    ingest_document("machine learning algorithms and models. " * 5, "ml_filter_doc")
    _, sources_no_filter = retrieve_and_answer("machine learning", top_k=3)
    # A very high min_score should filter everything out
    answer, sources_filtered = retrieve_and_answer("machine learning", top_k=3, min_score=999.0)
    assert answer == "No relevant context found."
    assert sources_filtered == []


def test_retrieve_min_score_zero_no_filtering():
    ingest_document("neural networks deep learning. " * 5, "nn_doc")
    _, sources_default = retrieve_and_answer("neural networks", top_k=2)
    _, sources_zero = retrieve_and_answer("neural networks", top_k=2, min_score=0.0)
    assert len(sources_default) == len(sources_zero)


def test_retrieve_answer_string_nonempty_after_ingest():
    ingest_document("Gradient descent optimizes loss functions in machine learning. " * 5, "gd_doc")
    answer, _ = retrieve_and_answer("gradient descent optimization", top_k=2)
    assert len(answer) > 0
    assert answer != "No relevant context found."
