import json
import os

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.features import generate_training_corpus

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
METRICS_PATH = os.getenv("METRICS_PATH", "metrics.json")


def train_model(X: np.ndarray = None, y: np.ndarray = None) -> tuple:
    if X is None or y is None:
        X, y = generate_training_corpus()

    clf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(clf_pipe, X, y, cv=cv, scoring="roc_auc")

    clf_pipe.fit(X, y)

    # IsolationForest trained on normal samples only for unsupervised anomaly detection
    X_normal = X[y == 0]
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(X_normal)

    bundle = {"classifier": clf_pipe, "isolation_forest": iso}
    joblib.dump(bundle, MODEL_PATH)

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
    if not os.path.exists(MODEL_PATH):
        bundle, _ = train_model()
        return bundle
    return joblib.load(MODEL_PATH)


def predict_anomaly(model_bundle: dict, features: np.ndarray) -> dict:
    clf = model_bundle["classifier"]
    iso = model_bundle["isolation_forest"]

    feat_2d = features.reshape(1, -1)

    clf_prob = clf.predict_proba(feat_2d)[0][1]
    iso_score = iso.decision_function(feat_2d)[0]
    iso_label = iso.predict(feat_2d)[0]  # -1 = anomaly, 1 = normal

    # Ensemble: anomaly if either detector flags it
    is_anomaly = bool(clf_prob > 0.5 or iso_label == -1)
    ensemble_score = float(clf_prob * 0.6 + (1.0 - (iso_score + 0.5)) * 0.4)

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": round(float(ensemble_score), 4),
        "classifier_prob": round(float(clf_prob), 4),
        "isolation_score": round(float(iso_score), 4),
    }
