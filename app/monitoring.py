import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.constants import DRIFT_P_VALUE_THRESHOLD
from app.database import DriftLog, PredictionLog

logger = logging.getLogger(__name__)


def compute_drift(reference: List[float], current: List[float]) -> dict:
    """Run a two-sample KS test to detect score distribution drift.

    Args:
        reference: Baseline anomaly score sample.
        current: Recent anomaly score sample.

    Returns:
        Dict with ks_statistic, p_value, and drift_detected flag.
    """
    if len(reference) < 2 or len(current) < 2:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False, "error": "insufficient_data"}
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < DRIFT_P_VALUE_THRESHOLD),
    }


def log_prediction(
    db: Session,
    query: str,
    anomaly_score: float,
    is_anomaly: bool,
    rag_used: bool = False,
    response_time_ms: Optional[float] = None,
) -> PredictionLog:
    """Persist a single prediction event to the database.

    Args:
        db: Active SQLAlchemy session.
        query: Original query text.
        anomaly_score: Ensemble anomaly score in [0, 1].
        is_anomaly: Whether the ensemble flagged this as anomalous.
        rag_used: Whether RAG retrieval was performed.
        response_time_ms: End-to-end response latency in milliseconds.

    Returns:
        The persisted PredictionLog ORM instance.
    """
    record = PredictionLog(
        query=query,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly,
        rag_context_used=rag_used,
        response_time_ms=response_time_ms,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def log_drift(db: Session, drift_result: dict, sample_size: int) -> DriftLog:
    """Persist a drift detection result to the database.

    Args:
        db: Active SQLAlchemy session.
        drift_result: Output of compute_drift().
        sample_size: Number of current-window samples used.

    Returns:
        The persisted DriftLog ORM instance.
    """
    record = DriftLog(
        ks_statistic=drift_result["ks_statistic"],
        p_value=drift_result["p_value"],
        drift_detected=drift_result["drift_detected"],
        sample_size=sample_size,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_recent_scores(db: Session, hours: int = 24) -> List[float]:
    """Query anomaly scores from the last N hours.

    Args:
        db: Active SQLAlchemy session.
        hours: Lookback window in hours.

    Returns:
        List of anomaly score floats.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = db.query(PredictionLog.anomaly_score).filter(PredictionLog.created_at >= since).all()
    return [r.anomaly_score for r in rows]


def _load_model_metrics(metrics_path: str) -> Dict[str, object]:
    """Load model metrics from a JSON file, returning an empty dict on failure."""
    if not os.path.exists(metrics_path):
        return {}
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read metrics file %s: %s", metrics_path, exc)
        return {}


def compute_score_percentiles(scores: List[float]) -> Dict[str, float]:
    """Compute p50, p90, and p99 percentiles of a score distribution.

    Args:
        scores: List of anomaly score floats.

    Returns:
        Dict with keys p50, p90, p99. Returns zeros when the list is empty.
    """
    if not scores:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
    arr = np.array(scores, dtype=float)
    return {
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
    }


def get_system_metrics(db: Session) -> dict:
    """Aggregate system health and model metrics from the database.

    Args:
        db: Active SQLAlchemy session.

    Returns:
        Dict with prediction counts, anomaly rate, drift state, and model AUC.
    """
    try:
        total = db.query(PredictionLog).count()
        anomalies = db.query(PredictionLog).filter(PredictionLog.is_anomaly).count()
        recent_scores = get_recent_scores(db, hours=1)
        last_drift = db.query(DriftLog).order_by(DriftLog.created_at.desc()).first()
    except Exception as exc:
        logger.error("Database query failed in get_system_metrics: %s", exc)
        total, anomalies, recent_scores, last_drift = 0, 0, [], None

    metrics_path = os.getenv("METRICS_PATH", "metrics.json")
    model_metrics = _load_model_metrics(metrics_path)

    return {
        "total_predictions": total,
        "total_anomalies": anomalies,
        "anomaly_rate": round(anomalies / max(total, 1), 4),
        "recent_1h_count": len(recent_scores),
        "recent_1h_avg_score": round(sum(recent_scores) / max(len(recent_scores), 1), 4),
        "last_drift_detected": last_drift.drift_detected if last_drift else None,
        "last_drift_ks": last_drift.ks_statistic if last_drift else None,
        "model_auc": model_metrics.get("auc_mean"),
        "model_n_features": model_metrics.get("n_features"),
    }
