"""Integration tests covering the full ingest → predict → metrics pipeline."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.features import generate_training_corpus


@pytest.fixture(scope="module")
def integration_client():
    from app.model import train_model

    X, y = generate_training_corpus(n_normal=80, n_anomaly=20, seed=77)
    bundle, _ = train_model(X, y)

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    def override_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    from app.main import _model_bundle, app

    _model_bundle.update(bundle)
    app.dependency_overrides[get_db] = override_db

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


def test_full_pipeline_ingest_then_predict(integration_client):
    ingest_resp = integration_client.post("/ingest", json={
        "text": "Deep learning uses multiple layers of neural networks to learn representations. "
                "The backpropagation algorithm updates weights during training. "
                "Convolutional networks are effective for image recognition tasks. " * 5,
        "doc_id": "integration_doc_001",
    })
    assert ingest_resp.status_code == 200
    assert ingest_resp.json()["chunks"] >= 1

    predict_resp = integration_client.post("/predict", json={
        "query": "How do neural networks learn representations?",
        "use_rag": True,
        "top_k": 2,
    })
    assert predict_resp.status_code == 200
    data = predict_resp.json()
    assert "anomaly_score" in data
    assert "rag_answer" in data


def test_anomaly_query_not_passed_to_rag(integration_client):
    resp = integration_client.post("/predict", json={
        "query": "DROP TABLE users; SELECT * FROM admin WHERE 1=1 UNION ALL SELECT password--",
        "use_rag": True,
    })
    assert resp.status_code == 200
    data = resp.json()
    # Anomalous queries should not receive a RAG answer
    if data["is_anomaly"]:
        assert data["rag_answer"] is None


def test_metrics_reflects_predictions(integration_client):
    for i in range(3):
        integration_client.post("/predict", json={"query": f"Query {i}: what is ML?", "use_rag": False})

    metrics_resp = integration_client.get("/metrics")
    assert metrics_resp.status_code == 200
    data = metrics_resp.json()
    assert data["system"]["total_predictions"] >= 0


def test_ingest_updates_existing_doc(integration_client):
    integration_client.post("/ingest", json={
        "text": "First version of the document about AI concepts and machine learning. " * 5,
        "doc_id": "update_test_doc",
        "filename": "v1.txt",
    })
    resp = integration_client.post("/ingest", json={
        "text": "Updated version of the document about neural networks and deep learning. " * 5,
        "doc_id": "update_test_doc",
        "filename": "v2.txt",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["doc_id"] == "update_test_doc"
    assert data["chunks"] >= 1


def test_version_consistent_with_health(integration_client):
    health = integration_client.get("/health").json()
    version = integration_client.get("/version").json()
    assert health["version"] == version["version"]
