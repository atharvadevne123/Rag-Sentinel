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


import pytest


@pytest.mark.parametrize("hours", [1, 6, 24, 72])
def test_get_recent_scores_different_windows(db_session, hours):
    scores = get_recent_scores(db_session, hours=hours)
    assert isinstance(scores, list)


def test_log_prediction_default_rag_false(db_session):
    record = log_prediction(db_session, "simple query", 0.1, False)
    assert record.rag_context_used is False


def test_log_prediction_with_response_time(db_session):
    record = log_prediction(db_session, "timed query", 0.5, True, response_time_ms=123.45)
    assert record.response_time_ms == pytest.approx(123.45)


def test_compute_drift_symmetric_distributions():
    import numpy as np
    ref = list(np.random.default_rng(10).normal(0, 1, 200))
    cur = list(np.random.default_rng(10).normal(0, 1, 200))
    result = compute_drift(ref, cur)
    assert result["p_value"] > 0.05


def test_get_system_metrics_zero_counts(db_session):
    metrics = get_system_metrics(db_session)
    assert metrics["anomaly_rate"] >= 0.0
    assert metrics["recent_1h_count"] >= 0


def test_log_drift_false_detection(db_session):
    drift = {"ks_statistic": 0.05, "p_value": 0.8, "drift_detected": False}
    record = log_drift(db_session, drift, sample_size=200)
    assert record.drift_detected is False
    assert record.ks_statistic == 0.05


def test_compute_drift_returns_all_keys():
    import numpy as np
    ref = list(np.random.default_rng(99).uniform(0, 1, 30))
    cur = list(np.random.default_rng(100).uniform(0, 1, 30))
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert "drift_detected" in result


def test_log_prediction_is_anomaly_true(db_session):
    record = log_prediction(db_session, "injection query", 0.9, True)
    assert record.is_anomaly is True
    assert record.anomaly_score == 0.9
