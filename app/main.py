import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
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

_model_bundle = {}
_query_history: list = []
_reference_scores: list = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    bundle = load_model()
    _model_bundle.update(bundle)
    yield


app = FastAPI(
    title="RAG Sentinel",
    description="RAG-powered document intelligence with ML anomaly detection and drift monitoring",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    use_rag: bool = Field(default=True)
    top_k: int = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    query: str
    is_anomaly: bool
    anomaly_score: float
    classifier_prob: float
    isolation_score: float
    rag_answer: Optional[str] = None
    rag_sources: Optional[list] = None
    response_time_ms: float


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=10)
    doc_id: str = Field(..., min_length=1, max_length=64)
    filename: str = Field(default="manual_input")


class RetrainResponse(BaseModel):
    status: str
    auc_mean: float
    auc_std: float
    n_features: int


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": len(_model_bundle) > 0,
        "version": "1.0.0",
    }


@app.post("/predict", response_model=QueryResponse)
def predict(req: QueryRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    t0 = time.time()

    features = extract_query_features(req.query, _query_history[-10:])
    result = predict_anomaly(_model_bundle, features)

    rag_answer = None
    rag_sources = None
    if req.use_rag and not result["is_anomaly"]:
        try:
            from rag.retriever import retrieve_and_answer
            rag_answer, rag_sources = retrieve_and_answer(req.query, top_k=req.top_k)
        except Exception:
            rag_answer = "RAG index not yet populated. Ingest documents first."
            rag_sources = []

    elapsed_ms = (time.time() - t0) * 1000
    _query_history.append(req.query)

    background_tasks.add_task(
        log_prediction,
        db=db,
        query=req.query,
        anomaly_score=result["anomaly_score"],
        is_anomaly=result["is_anomaly"],
        rag_used=req.use_rag,
        response_time_ms=round(elapsed_ms, 2),
    )

    return QueryResponse(
        query=req.query,
        is_anomaly=result["is_anomaly"],
        anomaly_score=result["anomaly_score"],
        classifier_prob=result["classifier_prob"],
        isolation_score=result["isolation_score"],
        rag_answer=rag_answer,
        rag_sources=rag_sources,
        response_time_ms=round(elapsed_ms, 2),
    )


@app.post("/ingest")
def ingest(req: IngestRequest, db: Session = Depends(get_db)):
    from app.database import DocumentIndex
    from rag.ingest import ingest_document
    try:
        chunk_count = ingest_document(req.text, req.doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

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
def metrics(db: Session = Depends(get_db)):
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


@app.post("/retrain", response_model=RetrainResponse)
def retrain(db: Session = Depends(get_db)):
    current_scores = get_recent_scores(db, hours=24)
    _reference_scores.clear()
    _reference_scores.extend(current_scores)

    bundle, metrics = train_model()
    _model_bundle.clear()
    _model_bundle.update(bundle)

    return RetrainResponse(
        status="retrained",
        auc_mean=metrics["auc_mean"],
        auc_std=metrics["auc_std"],
        n_features=metrics["n_features"],
    )
