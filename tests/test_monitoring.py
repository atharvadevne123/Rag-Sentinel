from app.monitoring import (
    compute_drift,
    get_recent_scores,
    get_system_metrics,
    log_drift,
    log_prediction,
)


def test_compute_drift_no_drift():
    import numpy as np
    ref = list(np.random.default_rng(0).uniform(0.1, 0.3, 50))
    cur = list(np.random.default_rng(1).uniform(0.1, 0.3, 50))
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert "drift_detected" in result
    assert result["drift_detected"] is False


def test_compute_drift_detects_drift():
    import numpy as np
    ref = list(np.random.default_rng(0).uniform(0.0, 0.2, 100))
    cur = list(np.random.default_rng(1).uniform(0.8, 1.0, 100))
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.5


def test_compute_drift_insufficient_data():
    result = compute_drift([0.5], [0.6])
    assert result["drift_detected"] is False
    assert "error" in result


def test_log_prediction(db_session):
    record = log_prediction(db_session, "test query", 0.25, False, rag_used=True, response_time_ms=42.0)
    assert record.id is not None
    assert record.query == "test query"
    assert record.anomaly_score == 0.25
    assert record.rag_context_used is True


def test_log_drift(db_session):
    drift = {"ks_statistic": 0.3, "p_value": 0.04, "drift_detected": True}
    record = log_drift(db_session, drift, sample_size=50)
    assert record.id is not None
    assert record.drift_detected is True
    assert record.sample_size == 50


def test_get_recent_scores_empty(db_session):
    scores = get_recent_scores(db_session, hours=1)
    assert isinstance(scores, list)


def test_get_system_metrics_structure(db_session):
    metrics = get_system_metrics(db_session)
    assert "total_predictions" in metrics
    assert "anomaly_rate" in metrics
    assert "recent_1h_count" in metrics
    assert "last_drift_detected" in metrics


def test_compute_drift_ks_statistic_range():
    import numpy as np
    ref = list(np.random.default_rng(42).normal(0, 1, 100))
    cur = list(np.random.default_rng(42).normal(5, 1, 100))
    result = compute_drift(ref, cur)
    assert 0.0 <= result["ks_statistic"] <= 1.0
