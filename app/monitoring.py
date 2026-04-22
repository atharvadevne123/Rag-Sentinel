import json
import os
from datetime import datetime, timedelta
from typing import List

from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftLog, PredictionLog


def compute_drift(reference: List[float], current: List[float]) -> dict:
    if len(reference) < 2 or len(current) < 2:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False, "error": "insufficient_data"}
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < 0.05),
    }


def log_prediction(db: Session, query: str, anomaly_score: float, is_anomaly: bool,
                   rag_used: bool = False, response_time_ms: float = None) -> PredictionLog:
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
    since = datetime.utcnow() - timedelta(hours=hours)
    rows = db.query(PredictionLog.anomaly_score).filter(PredictionLog.created_at >= since).all()
    return [r.anomaly_score for r in rows]


def get_system_metrics(db: Session) -> dict:
    total = db.query(PredictionLog).count()
    anomalies = db.query(PredictionLog).filter(PredictionLog.is_anomaly).count()
    recent_scores = get_recent_scores(db, hours=1)
    last_drift = db.query(DriftLog).order_by(DriftLog.created_at.desc()).first()

    metrics_path = os.getenv("METRICS_PATH", "metrics.json")
    model_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            model_metrics = json.load(f)

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
