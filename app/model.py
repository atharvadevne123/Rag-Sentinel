import json
import logging
import os
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.constants import (
    ANOMALY_CLASSIFIER_THRESHOLD,
    ANOMALY_ENSEMBLE_CLASSIFIER_WEIGHT,
    ANOMALY_ENSEMBLE_ISOLATION_WEIGHT,
)
from app.features import generate_training_corpus

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
METRICS_PATH = os.getenv("METRICS_PATH", "metrics.json")


def train_model(
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
) -> Tuple[dict, dict]:
    """Train a RandomForest + IsolationForest ensemble and persist it to disk.

    If X and y are not supplied the function generates a synthetic training
    corpus via generate_training_corpus().

    Args:
        X: Feature matrix of shape (n_samples, n_features). Optional.
        y: Binary label vector (0=normal, 1=anomaly). Optional.

    Returns:
        Tuple of (model_bundle, metrics) where model_bundle has keys
        'classifier' and 'isolation_forest', and metrics contains AUC stats.
    """
    if X is None or y is None:
        logger.info("No training data provided; generating synthetic corpus.")
        X, y = generate_training_corpus()

    clf_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=6,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(clf_pipe, X, y, cv=cv, scoring="roc_auc")
    logger.info("Cross-val AUC: %.4f ± %.4f", auc_scores.mean(), auc_scores.std())

    clf_pipe.fit(X, y)

    X_normal = X[y == 0]
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(X_normal)

    bundle = {"classifier": clf_pipe, "isolation_forest": iso}
    joblib.dump(bundle, MODEL_PATH)
    logger.info("Model bundle saved to %s", MODEL_PATH)

    metrics = {
        "auc_mean": round(float(auc_scores.mean()), 4),
        "auc_std": round(float(auc_scores.std()), 4),
        "n_features": X.shape[1],
        "n_train_samples": len(X),
        "n_anomaly_samples": int(y.sum()),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return bundle, metrics


def load_model() -> dict:
    """Load the persisted model bundle from disk, training if absent.

    Returns:
        Dict with keys 'classifier' (sklearn Pipeline) and
        'isolation_forest' (IsolationForest).
    """
    if not os.path.exists(MODEL_PATH):
        logger.info("No model file found at %s; training from scratch.", MODEL_PATH)
        bundle, _ = train_model()
        return bundle
    logger.info("Loading model bundle from %s", MODEL_PATH)
    return joblib.load(MODEL_PATH)


def predict_anomaly(model_bundle: dict, features: np.ndarray) -> dict:
    """Score a single query feature vector using the ensemble.

    The ensemble combines a RandomForest classifier probability (weight 0.6)
    with an IsolationForest score (weight 0.4). A prediction is flagged as
    anomalous if either sub-model triggers.

    Args:
        model_bundle: Dict returned by train_model() or load_model().
        features: 1-D float32 feature vector from extract_query_features().

    Returns:
        Dict with keys is_anomaly, anomaly_score, classifier_prob, and
        isolation_score.
    """
    clf = model_bundle["classifier"]
    iso = model_bundle["isolation_forest"]

    feat_2d = features.reshape(1, -1)

    expected_n_features = clf.n_features_in_
    if feat_2d.shape[1] != expected_n_features:
        raise ValueError(
            f"Feature count mismatch: model expects {expected_n_features}, got {feat_2d.shape[1]}"
        )

    clf_prob = float(clf.predict_proba(feat_2d)[0][1])
    iso_score = float(iso.decision_function(feat_2d)[0])
    iso_label = int(iso.predict(feat_2d)[0])  # -1 = anomaly, 1 = normal

    is_anomaly = bool(clf_prob > ANOMALY_CLASSIFIER_THRESHOLD or iso_label == -1)
    ensemble_score = float(
        clf_prob * ANOMALY_ENSEMBLE_CLASSIFIER_WEIGHT + (1.0 - (iso_score + 0.5)) * ANOMALY_ENSEMBLE_ISOLATION_WEIGHT
    )

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": round(ensemble_score, 4),
        "classifier_prob": round(clf_prob, 4),
        "isolation_score": round(iso_score, 4),
    }
