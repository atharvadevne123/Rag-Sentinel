"""
Airflow DAG: automated daily retraining pipeline for RAG Sentinel.
Pulls recent prediction logs, checks drift, retrains if needed.
"""
from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

import os
import json
import logging

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "rag-sentinel",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_drift(**context):
    """Pull last 24h scores from DB and run KS drift test."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database import PredictionLog, DriftLog, Base
    from app.monitoring import compute_drift, get_recent_scores

    db_url = os.getenv("DATABASE_URL", "sqlite:///./rag_sentinel.db")
    engine = create_engine(db_url, connect_args={"check_same_thread": False} if "sqlite" in db_url else {})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        recent = get_recent_scores(db, hours=24)
        reference = get_recent_scores(db, hours=72)

        if len(recent) < 10 or len(reference) < 10:
            logger.info("Insufficient data for drift check. Skipping.")
            return {"drift_detected": False}

        result = compute_drift(reference, recent)
        logger.info(f"Drift result: {result}")

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


def retrain_if_needed(**context):
    """Retrain model if drift detected or scheduled weekly."""
    drift_detected = context["task_instance"].xcom_pull(key="drift_detected", task_ids="check_drift")

    force_retrain = datetime.utcnow().weekday() == 6  # Always retrain on Sunday
    should_retrain = drift_detected or force_retrain

    if not should_retrain:
        logger.info("No drift detected and not weekly retrain day. Skipping retrain.")
        return {"retrained": False}

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from app.model import train_model

    logger.info("Retraining model...")
    _, metrics = train_model()
    logger.info(f"Retrain complete. Metrics: {metrics}")

    with open("retrain_log.json", "a") as f:
        entry = {"timestamp": datetime.utcnow().isoformat(), "metrics": metrics, "reason": "drift" if drift_detected else "scheduled"}
        f.write(json.dumps(entry) + "\n")

    return {"retrained": True, "metrics": metrics}


def validate_model(**context):
    """Sanity-check the freshly trained model."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import numpy as np
    from app.model import load_model, predict_anomaly
    from app.features import extract_query_features

    bundle = load_model()

    test_queries = [
        ("What is machine learning?", False),
        ("DROP TABLE users;--", True),
    ]
    passed = 0
    for q, expected_anomaly in test_queries:
        features = extract_query_features(q, [])
        result = predict_anomaly(bundle, features)
        if result["is_anomaly"] == expected_anomaly:
            passed += 1

    logger.info(f"Validation: {passed}/{len(test_queries)} checks passed")
    if passed < len(test_queries):
        raise ValueError(f"Model validation failed: only {passed}/{len(test_queries)} passed")
    return {"validation_passed": True}


if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="rag_sentinel_retrain",
        default_args=DEFAULT_ARGS,
        description="Daily drift check and conditional retraining for RAG Sentinel",
        schedule_interval="0 2 * * *",  # 2 AM UTC daily
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
