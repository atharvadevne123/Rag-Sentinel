"""Performance and caching tests for RAG Sentinel."""

from __future__ import annotations

import time

import pytest


def test_regex_cache_returns_same_object():
    from app.features import _bracket_pattern, _sql_pattern

    p1 = _sql_pattern()
    p2 = _sql_pattern()
    assert p1 is p2

    b1 = _bracket_pattern()
    b2 = _bracket_pattern()
    assert b1 is b2


def test_get_settings_returns_cached_instance():
    from app.config import get_settings

    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_extract_features_speed():
    from app.features import extract_query_features

    queries = ["What is machine learning?"] * 100
    t0 = time.perf_counter()
    for q in queries:
        extract_query_features(q, [])
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"Feature extraction too slow: {elapsed:.2f}s for 100 queries"


def test_embed_speed():
    from rag.index import SentinelIndex

    idx = SentinelIndex()
    texts = ["machine learning is powerful"] * 50
    t0 = time.perf_counter()
    idx.embed(texts)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"Embedding too slow: {elapsed:.2f}s for 50 texts"


@pytest.mark.parametrize("n_queries", [1, 10, 50])
def test_batch_feature_extraction_scales(n_queries):
    from app.features import extract_query_features

    queries = ["test query about machine learning"] * n_queries
    history = []
    t0 = time.perf_counter()
    for q in queries:
        extract_query_features(q, history[-5:])
        history.append(q)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0
