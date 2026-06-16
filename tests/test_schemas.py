"""Tests for app/schemas.py — Pydantic request/response models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    HealthResponse,
    IndexStatsResponse,
    IngestResponse,
    PredictionRequest,
    PredictionResponse,
    RetrainResponse,
    VersionResponse,
)


# ---------------------------------------------------------------------------
# HealthResponse
# ---------------------------------------------------------------------------


def test_health_response_valid():
    h = HealthResponse(status="ok", model_loaded=True, database_ok=True, version="1.1.0")
    assert h.status == "ok"
    assert h.model_loaded is True
    assert h.database_ok is True
    assert h.version == "1.1.0"


def test_health_response_missing_field_raises():
    with pytest.raises(ValidationError):
        HealthResponse(status="ok", model_loaded=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# VersionResponse
# ---------------------------------------------------------------------------


def test_version_response_valid():
    v = VersionResponse(version="1.1.0")
    assert v.version == "1.1.0"


def test_version_response_missing_raises():
    with pytest.raises(ValidationError):
        VersionResponse()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# PredictionRequest
# ---------------------------------------------------------------------------


def test_prediction_request_defaults():
    req = PredictionRequest(query="What is AI?")
    assert req.use_rag is True
    assert req.top_k == 3


def test_prediction_request_custom_values():
    req = PredictionRequest(query="test query", use_rag=False, top_k=5)
    assert req.use_rag is False
    assert req.top_k == 5


def test_prediction_request_empty_query_raises():
    with pytest.raises(ValidationError):
        PredictionRequest(query="")


def test_prediction_request_too_long_query_raises():
    with pytest.raises(ValidationError):
        PredictionRequest(query="x" * 2001)


def test_prediction_request_top_k_zero_raises():
    with pytest.raises(ValidationError):
        PredictionRequest(query="hello", top_k=0)


def test_prediction_request_top_k_eleven_raises():
    with pytest.raises(ValidationError):
        PredictionRequest(query="hello", top_k=11)


def test_prediction_request_top_k_boundary_valid():
    assert PredictionRequest(query="hello", top_k=1).top_k == 1
    assert PredictionRequest(query="hello", top_k=10).top_k == 10


# ---------------------------------------------------------------------------
# PredictionResponse
# ---------------------------------------------------------------------------


def test_prediction_response_required_fields():
    resp = PredictionResponse(
        query="test",
        is_anomaly=False,
        anomaly_score=0.1,
        classifier_prob=0.2,
        isolation_score=0.3,
        response_time_ms=42.0,
    )
    assert resp.query == "test"
    assert resp.rag_answer is None
    assert resp.rag_sources is None


def test_prediction_response_with_rag_fields():
    resp = PredictionResponse(
        query="test",
        is_anomaly=True,
        anomaly_score=0.9,
        classifier_prob=0.85,
        isolation_score=0.7,
        response_time_ms=10.0,
        rag_answer="Some answer",
        rag_sources=[{"doc_id": "doc1", "score": 0.9}],
    )
    assert resp.rag_answer == "Some answer"
    assert resp.rag_sources[0]["doc_id"] == "doc1"


# ---------------------------------------------------------------------------
# IngestResponse
# ---------------------------------------------------------------------------


def test_ingest_response_valid():
    r = IngestResponse(status="ingested", doc_id="doc_001", chunks=5)
    assert r.status == "ingested"
    assert r.chunks == 5


def test_ingest_response_missing_field_raises():
    with pytest.raises(ValidationError):
        IngestResponse(status="ingested", doc_id="doc_001")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# IndexStatsResponse
# ---------------------------------------------------------------------------


def test_index_stats_response_valid():
    s = IndexStatsResponse(chunk_count=10, doc_count=2, faiss_available=False, faiss_total=0)
    assert s.chunk_count == 10
    assert s.faiss_available is False


# ---------------------------------------------------------------------------
# RetrainResponse
# ---------------------------------------------------------------------------


def test_retrain_response_valid():
    r = RetrainResponse(status="retrained", auc_mean=0.92, auc_std=0.03, n_features=15)
    assert r.n_features == 15
    assert r.auc_mean == pytest.approx(0.92)


def test_retrain_response_missing_auc_raises():
    with pytest.raises(ValidationError):
        RetrainResponse(status="retrained", auc_std=0.03, n_features=15)  # type: ignore[call-arg]
