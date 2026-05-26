"""Airflow DAG: automated daily retraining pipeline for RAG Sentinel.

Pulls recent prediction logs, checks drift, retrains if needed, and
validates the newly trained model before notifying completion.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import PythonOperator

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

DEFAULT_ARGS: Dict[str, Any] = {
    "owner": "rag-sentinel",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_drift(**context: Any) -> Dict[str, Any]:
    """Pull last 24 h scores from the database and run a KS drift test.

    Pushes 'drift_detected' (bool) to XCom for downstream tasks.

    Args:
        **context: Airflow task context dictionary.

    Returns:
        Dict with keys ks_statistic, p_value, and drift_detected.
    """
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.database import Base, DriftLog
    from app.monitoring import compute_drift, get_recent_scores

    db_url = os.getenv("DATABASE_URL", "sqlite:///./rag_sentinel.db")
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        recent = get_recent_scores(db, hours=24)
        reference = get_recent_scores(db, hours=72)

        if len(recent) < 10 or len(reference) < 10:
            logger.info("Insufficient data for drift check (recent=%d, ref=%d). Skipping.", len(recent), len(reference))
            return {"drift_detected": False}

        result = compute_drift(reference, recent)
        logger.info("Drift result: %s", result)

        drift_log = DriftLog(
            ks_statistic=result["ks_statistic"],
            p_value=result["p_value"],
            drift_detected=result["drift_detected"],
            sample_size=len(recent),
        )
        db.add(drift_log)
        db.commit()

        context["task_instance"].xcom_push(key="drift_detected", value=result["drift_detected"])
        return result
    finally:
        db.close()


def retrain_if_needed(**context: Any) -> Dict[str, Any]:
    """Retrain the model if drift was detected or it is a scheduled Sunday retrain.

    Args:
        **context: Airflow task context dictionary.

    Returns:
        Dict with 'retrained' flag and optional 'metrics' dict.
    """
    drift_detected: bool = context["task_instance"].xcom_pull(
        key="drift_detected", task_ids="check_drift"
    )
    force_retrain = datetime.utcnow().weekday() == 6  # Always retrain on Sunday
    should_retrain = drift_detected or force_retrain

    if not should_retrain:
        logger.info("No drift detected and not weekly retrain day. Skipping retrain.")
        return {"retrained": False}

    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from app.model import train_model

    logger.info("Retraining model (drift=%s, forced=%s)...", drift_detected, force_retrain)
    _, metrics = train_model()
    logger.info("Retrain complete. Metrics: %s", metrics)

    with open("retrain_log.json", "a") as f:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "reason": "drift" if drift_detected else "scheduled",
        }
        f.write(json.dumps(entry) + "\n")

    return {"retrained": True, "metrics": metrics}


def validate_model(**context: Any) -> Dict[str, Any]:
    """Sanity-check the freshly trained model against known good/bad queries.

    Args:
        **context: Airflow task context dictionary.

    Returns:
        Dict with 'validation_passed' boolean.

    Raises:
        ValueError: If fewer than all validation checks pass.
    """
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from app.features import extract_query_features
    from app.model import load_model, predict_anomaly

    bundle = load_model()

    test_cases = [
        ("What is machine learning?", False),
        ("DROP TABLE users;--", True),
    ]
    passed = sum(
        1
        for query, expected in test_cases
        if predict_anomaly(bundle, extract_query_features(query, []))["is_anomaly"] == expected
    )

    logger.info("Validation: %d/%d checks passed", passed, len(test_cases))
    if passed < len(test_cases):
        raise ValueError(f"Model validation failed: only {passed}/{len(test_cases)} passed")
    return {"validation_passed": True}


if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="rag_sentinel_retrain",
        default_args=DEFAULT_ARGS,
        description="Daily drift check and conditional retraining for RAG Sentinel",
        schedule_interval="0 2 * * *",
        catchup=False,
        tags=["ml", "monitoring", "retrain"],
    ) as dag:
        t_check_drift = PythonOperator(
            task_id="check_drift",
            python_callable=check_drift,
            provide_context=True,
        )
        t_retrain = PythonOperator(
            task_id="retrain_if_needed",
            python_callable=retrain_if_needed,
            provide_context=True,
        )
        t_validate = PythonOperator(
            task_id="validate_model",
            python_callable=validate_model,
            provide_context=True,
        )
        t_notify = BashOperator(
            task_id="notify_complete",
            bash_command='echo "RAG Sentinel retrain pipeline complete at $(date)"',
        )

        t_check_drift >> t_retrain >> t_validate >> t_notify
