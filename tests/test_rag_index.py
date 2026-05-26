from __future__ import annotations

import numpy as np
import pytest

from rag.index import EMBED_DIM, SentinelIndex, get_index, reset_index


@pytest.fixture(autouse=True)
def fresh_index():
    reset_index()
    yield
    reset_index()


def test_embed_output_shape():
    idx = SentinelIndex()
    vecs = idx.embed(["Hello world", "Second string"])
    assert vecs.shape == (2, EMBED_DIM)
    assert vecs.dtype == np.float32


def test_embed_normalised():
    idx = SentinelIndex()
    vecs = idx.embed(["Hello world"])
    norm = float(np.linalg.norm(vecs[0]))
    assert abs(norm - 1.0) < 1e-5


def test_embed_empty_string_returns_zero_vector():
    idx = SentinelIndex()
    vecs = idx.embed([""])
    assert vecs.shape == (1, EMBED_DIM)
    assert float(np.linalg.norm(vecs[0])) == pytest.approx(0.0)


def test_add_and_len():
    idx = SentinelIndex()
    vecs = idx.embed(["chunk one", "chunk two"])
    idx.add(vecs, ["chunk one", "chunk two"], "doc_a")
    assert len(idx) == 2


def test_search_returns_top_k():
    idx = SentinelIndex()
    chunks = ["machine learning basics", "deep neural networks", "random forests"]
    vecs = idx.embed(chunks)
    idx.add(vecs, chunks, "doc_x")
    q_vec = idx.embed(["machine learning"])[0]
    results = idx.search(q_vec, top_k=2)
    assert len(results) == 2
    assert all(len(r) == 3 for r in results)


def test_search_empty_index_returns_empty():
    idx = SentinelIndex()
    q_vec = idx.embed(["test"])[0]
    results = idx.search(q_vec, top_k=3)
    assert results == []


def test_get_index_singleton():
    idx1 = get_index()
    idx2 = get_index()
    assert idx1 is idx2


def test_reset_index_creates_new_instance():
    idx1 = get_index()
    reset_index()
    idx2 = get_index()
    assert idx1 is not idx2


def test_repr():
    idx = SentinelIndex()
    r = repr(idx)
    assert "SentinelIndex" in r
    assert "chunks=0" in r


def test_search_score_ordering():
    idx = SentinelIndex()
    chunks = ["machine learning is great", "cooking recipes", "deep learning tutorial"]
    vecs = idx.embed(chunks)
    idx.add(vecs, chunks, "doc_y")
    q_vec = idx.embed(["machine learning deep learning"])[0]
    results = idx.search(q_vec, top_k=3)
    scores = [r[2] for r in results]
    assert scores == sorted(scores, reverse=True)
