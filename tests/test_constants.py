from __future__ import annotations

from app.constants import (
    ANOMALY_CLASSIFIER_THRESHOLD,
    ANOMALY_ENSEMBLE_CLASSIFIER_WEIGHT,
    ANOMALY_ENSEMBLE_ISOLATION_WEIGHT,
    APP_NAME,
    APP_VERSION,
    DEFAULT_TOP_K,
    DRIFT_MIN_SAMPLE_SIZE,
    DRIFT_P_VALUE_THRESHOLD,
    FEATURE_ROLLING_WINDOW,
    QUERY_HISTORY_WINDOW,
)


def test_ensemble_weights_sum_to_one():
    assert abs(ANOMALY_ENSEMBLE_CLASSIFIER_WEIGHT + ANOMALY_ENSEMBLE_ISOLATION_WEIGHT - 1.0) < 1e-9


def test_anomaly_threshold_in_valid_range():
    assert 0.0 < ANOMALY_CLASSIFIER_THRESHOLD < 1.0


def test_drift_p_value_threshold_in_valid_range():
    assert 0.0 < DRIFT_P_VALUE_THRESHOLD < 1.0


def test_drift_min_sample_size_positive():
    assert DRIFT_MIN_SAMPLE_SIZE > 1


def test_default_top_k_positive():
    assert DEFAULT_TOP_K >= 1


def test_query_history_window_positive():
    assert QUERY_HISTORY_WINDOW > 0


def test_feature_rolling_window_positive():
    assert FEATURE_ROLLING_WINDOW > 0


def test_app_name_non_empty():
    assert len(APP_NAME) > 0


def test_app_version_format():
    parts = APP_VERSION.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
