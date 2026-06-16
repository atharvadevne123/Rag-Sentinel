"""Pydantic response and request schemas shared across the app.

Centralising schemas here avoids circular imports and makes the API
contract easy to inspect without reading main.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    model_loaded: bool
    database_ok: bool
    version: str


class VersionResponse(BaseModel):
    """Response schema for GET /version."""

    version: str


class PredictionRequest(BaseModel):
    """Request schema for POST /predict."""

    query: str = Field(..., min_length=1, max_length=2000)
    use_rag: bool = Field(default=True)
    top_k: int = Field(default=3, ge=1, le=10)


class PredictionResponse(BaseModel):
    """Response schema for POST /predict."""

    query: str
    is_anomaly: bool
    anomaly_score: float
    classifier_prob: float
    isolation_score: float
    rag_answer: Optional[str] = None
    rag_sources: Optional[List[Dict[str, Any]]] = None
    response_time_ms: float


class IngestResponse(BaseModel):
    """Response schema for POST /ingest."""

    status: str
    doc_id: str
    chunks: int


class IndexStatsResponse(BaseModel):
    """Response schema for GET /index/stats."""

    chunk_count: int
    doc_count: int
    faiss_available: bool
    faiss_total: int


class RetrainResponse(BaseModel):
    """Response schema for POST /retrain."""

    status: str
    auc_mean: float
    auc_std: float
    n_features: int
