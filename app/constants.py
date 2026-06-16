"""Shared constants for RAG Sentinel.

Centralising these avoids magic numbers scattered across modules and makes
threshold tuning straightforward.
"""

from __future__ import annotations

from typing import Final

# Anomaly detection thresholds
ANOMALY_CLASSIFIER_THRESHOLD: Final[float] = 0.5
ANOMALY_ENSEMBLE_CLASSIFIER_WEIGHT: Final[float] = 0.6
ANOMALY_ENSEMBLE_ISOLATION_WEIGHT: Final[float] = 0.4

# Drift detection
DRIFT_P_VALUE_THRESHOLD: Final[float] = 0.05
DRIFT_MIN_SAMPLE_SIZE: Final[int] = 10

# Query history window for lag features
QUERY_HISTORY_WINDOW: Final[int] = 10
FEATURE_ROLLING_WINDOW: Final[int] = 5

# Feature vector length (must match extract_query_features output)
N_FEATURES: Final[int] = 15

# RAG index parameters
DEFAULT_TOP_K: Final[int] = 3

# Application metadata
APP_NAME: Final[str] = "RAG Sentinel"
APP_VERSION: Final[str] = "1.1.0"
