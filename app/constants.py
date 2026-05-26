"""Shared constants for RAG Sentinel.

Centralising these avoids magic numbers scattered across modules and makes
threshold tuning straightforward.
"""

from __future__ import annotations

# Anomaly detection thresholds
ANOMALY_CLASSIFIER_THRESHOLD: float = 0.5
ANOMALY_ENSEMBLE_CLASSIFIER_WEIGHT: float = 0.6
ANOMALY_ENSEMBLE_ISOLATION_WEIGHT: float = 0.4

# Drift detection
DRIFT_P_VALUE_THRESHOLD: float = 0.05
DRIFT_MIN_SAMPLE_SIZE: int = 10

# Query history window for lag features
QUERY_HISTORY_WINDOW: int = 10
FEATURE_ROLLING_WINDOW: int = 5

# RAG index parameters
DEFAULT_TOP_K: int = 3

# Application metadata
APP_NAME: str = "RAG Sentinel"
APP_VERSION: str = "1.0.0"
