import pytest

from app.monitoring import (
    compute_drift,
    compute_score_percentiles,
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


@pytest.mark.parametrize(
    "ref_mean,cur_mean,expect_drift",
    [
        (0.1, 0.9, True),
        (0.5, 0.5, False),
        (0.5, 0.52, False),
    ],
)
def test_compute_drift_parametrized_means(ref_mean, cur_mean, expect_drift):
    import numpy as np

    rng = np.random.default_rng(7)
    ref = list(rng.normal(ref_mean, 0.3, 100))
    cur = list(rng.normal(cur_mean, 0.3, 100))
    result = compute_drift(ref, cur)
    assert result["drift_detected"] == expect_drift


def test_get_system_metrics_model_auc_none_when_no_file(db_session, monkeypatch):

    monkeypatch.setenv("METRICS_PATH", "/nonexistent/metrics.json")
    metrics = get_system_metrics(db_session)
    assert metrics["model_auc"] is None


def test_get_system_metrics_returns_total_after_log(db_session):
    log_prediction(db_session, "q1", 0.1, False)
    log_prediction(db_session, "q2", 0.8, True)
    metrics = get_system_metrics(db_session)
    assert metrics["total_predictions"] >= 2
    assert metrics["total_anomalies"] >= 1


def test_compute_drift_empty_lists():
    result = compute_drift([], [])
    assert result["drift_detected"] is False
    assert "error" in result


def test_log_drift_and_retrieve(db_session):
    from app.database import DriftLog

    drift = {"ks_statistic": 0.42, "p_value": 0.001, "drift_detected": True}
    record = log_drift(db_session, drift, sample_size=75)
    assert record.id is not None
    retrieved = db_session.query(DriftLog).filter(DriftLog.id == record.id).first()
    assert retrieved is not None
    assert retrieved.ks_statistic == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# compute_score_percentiles
# ---------------------------------------------------------------------------


def test_compute_score_percentiles_empty():
    result = compute_score_percentiles([])
    assert result == {"p50": 0.0, "p90": 0.0, "p99": 0.0}


def test_compute_score_percentiles_single_value():
    result = compute_score_percentiles([0.5])
    assert result["p50"] == pytest.approx(0.5)
    assert result["p90"] == pytest.approx(0.5)
    assert result["p99"] == pytest.approx(0.5)


def test_compute_score_percentiles_uniform():
    scores = [float(i) / 100 for i in range(101)]
    result = compute_score_percentiles(scores)
    assert result["p50"] == pytest.approx(0.5, abs=0.01)
    assert result["p90"] == pytest.approx(0.9, abs=0.01)
    assert result["p99"] == pytest.approx(0.99, abs=0.01)


def test_compute_score_percentiles_keys():
    result = compute_score_percentiles([0.1, 0.5, 0.9])
    assert set(result.keys()) == {"p50", "p90", "p99"}


def test_compute_score_percentiles_all_same():
    result = compute_score_percentiles([0.3, 0.3, 0.3])
    assert result["p50"] == pytest.approx(0.3)
    assert result["p90"] == pytest.approx(0.3)
