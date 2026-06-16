import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.features import extract_query_features
from app.model import load_model, predict_anomaly, train_model
from app.monitoring import (
    compute_drift,
    get_recent_scores,
    get_system_metrics,
    log_drift,
    log_prediction,
)

logger = logging.getLogger(__name__)

_model_bundle: dict = {}
_query_history: List[str] = []
_reference_scores: List[float] = []

APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    bundle = load_model()
    _model_bundle.update(bundle)
    logger.info("RAG Sentinel started — model loaded, DB initialised.")
    yield
    logger.info("RAG Sentinel shutting down.")


app = FastAPI(
    title="RAG Sentinel",
    description="RAG-powered document intelligence with ML anomaly detection and drift monitoring",
    version=APP_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next) -> Response:
    """Attach a unique X-Request-ID header to every response for tracing."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next) -> Response:
    """Log method, path, status code, and elapsed time for every request."""
    t0 = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "%s %s -> %d  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


def _sanitize_query(text: str) -> str:
    """Strip null bytes and ASCII control characters (except tab/newline) from input."""
    return "".join(ch for ch in text if ch >= " " or ch in "\t\n")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    use_rag: bool = Field(default=True)
    top_k: int = Field(default=3, ge=1, le=10)

    model_config = {
        "json_schema_extra": {"examples": [{"query": "What is machine learning?", "use_rag": True, "top_k": 3}]}
    }


class QueryResponse(BaseModel):
    query: str
    is_anomaly: bool
    anomaly_score: float
    classifier_prob: float
    isolation_score: float
    rag_answer: Optional[str] = None
    rag_sources: Optional[List] = None
    response_time_ms: float


_MAX_INGEST_BYTES = int(os.getenv("MAX_INGEST_BYTES", str(500_000)))


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=_MAX_INGEST_BYTES)
    doc_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-\.]+$")
    filename: str = Field(default="manual_input", max_length=256)

    model_config = {
        "json_schema_extra": {"examples": [{"text": "Machine learning is a branch of AI.", "doc_id": "doc_001"}]}
    }


class RetrainResponse(BaseModel):
    status: str
    auc_mean: float
    auc_std: float
    n_features: int


def _get_rag_context(query: str, top_k: int) -> tuple:
    """Attempt RAG retrieval; return (answer, sources) or fallback strings."""
    try:
        from rag.retriever import retrieve_and_answer

        return retrieve_and_answer(query, top_k=top_k)
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        return "RAG index not yet populated. Ingest documents first.", []


@app.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    """Return service liveness, model-loaded status, and database connectivity."""
    db_ok = False
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception as exc:
        logger.warning("Database health check failed: %s", exc)

    return {
        "status": "ok",
        "model_loaded": len(_model_bundle) > 0,
        "database_ok": db_ok,
        "version": APP_VERSION,
    }


@app.get("/version")
def version() -> dict:
    """Return the current application version string."""
    return {"version": APP_VERSION}


@app.post("/predict", response_model=QueryResponse)
def predict(
    req: QueryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> QueryResponse:
    """Score a query for anomaly patterns and optionally retrieve a RAG answer.

    Args:
        req: Query request payload.
        background_tasks: FastAPI background task queue.
        db: Injected database session.

    Returns:
        QueryResponse with anomaly scores and optional RAG answer.
    """
    t0 = time.time()
    sanitized_query = _sanitize_query(req.query)

    features = extract_query_features(sanitized_query, _query_history[-10:])
    result = predict_anomaly(_model_bundle, features)

    rag_answer: Optional[str] = None
    rag_sources: Optional[List] = None
    if req.use_rag and not result["is_anomaly"]:
        rag_answer, rag_sources = _get_rag_context(sanitized_query, req.top_k)

    elapsed_ms = (time.time() - t0) * 1000
    _query_history.append(sanitized_query)

    background_tasks.add_task(
        log_prediction,
        db=db,
        query=sanitized_query,
        anomaly_score=result["anomaly_score"],
        is_anomaly=result["is_anomaly"],
        rag_used=req.use_rag,
        response_time_ms=round(elapsed_ms, 2),
    )

    return QueryResponse(
        query=sanitized_query,
        is_anomaly=result["is_anomaly"],
        anomaly_score=result["anomaly_score"],
        classifier_prob=result["classifier_prob"],
        isolation_score=result["isolation_score"],
        rag_answer=rag_answer,
        rag_sources=rag_sources,
        response_time_ms=round(elapsed_ms, 2),
    )


@app.post("/ingest")
def ingest(req: IngestRequest, db: Session = Depends(get_db)) -> dict:
    """Ingest a document into the RAG index and record it in the database.

    Args:
        req: Ingest request with text, doc_id, and filename.
        db: Injected database session.

    Returns:
        Dict with status, doc_id, and chunk count.
    """
    from app.database import DocumentIndex
    from rag.ingest import ingest_document

    try:
        chunk_count = ingest_document(req.text, req.doc_id)
    except Exception as exc:
        logger.error("Ingestion failed for doc_id=%s: %s", req.doc_id, exc)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    existing = db.query(DocumentIndex).filter(DocumentIndex.doc_id == req.doc_id).first()
    if existing:
        existing.chunk_count = chunk_count
        existing.filename = req.filename
    else:
        doc = DocumentIndex(doc_id=req.doc_id, filename=req.filename, chunk_count=chunk_count)
        db.add(doc)
    db.commit()

    return {"status": "ingested", "doc_id": req.doc_id, "chunks": chunk_count}


@app.get("/metrics")
def metrics(db: Session = Depends(get_db)) -> dict:
    """Return system health metrics and optional drift detection results."""
    sys_metrics = get_system_metrics(db)

    recent = get_recent_scores(db, hours=24)
    drift_result = None
    if _reference_scores and len(recent) >= 10:
        drift_result = compute_drift(_reference_scores, recent)
        log_drift(db, drift_result, sample_size=len(recent))

    return {
        "system": sys_metrics,
        "drift": drift_result,
    }


@app.get("/index/stats")
def index_stats() -> dict:
    """Return statistics about the in-memory RAG document index."""
    from rag.index import get_index

    idx = get_index()
    return idx.stats()


@app.post("/retrain", response_model=RetrainResponse)
def retrain(db: Session = Depends(get_db)) -> RetrainResponse:
    """Trigger an in-process model retrain using the latest 24-hour scores as reference.

    Args:
        db: Injected database session.

    Returns:
        RetrainResponse with new AUC metrics and feature count.
    """
    current_scores = get_recent_scores(db, hours=24)
    _reference_scores.clear()
    _reference_scores.extend(current_scores)

    bundle, train_metrics = train_model()
    _model_bundle.clear()
    _model_bundle.update(bundle)
    logger.info("Model retrained: AUC=%.4f ± %.4f", train_metrics["auc_mean"], train_metrics["auc_std"])

    return RetrainResponse(
        status="retrained",
        auc_mean=train_metrics["auc_mean"],
        auc_std=train_metrics["auc_std"],
        n_features=train_metrics["n_features"],
    )
