import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.features import generate_training_corpus


@pytest.fixture(scope="module")
def client():
    from app.model import train_model
    X, y = generate_training_corpus(n_normal=80, n_anomaly=20, seed=99)
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


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_normal_query(client):
    resp = client.post("/predict", json={"query": "What is machine learning?", "use_rag": False})
    assert resp.status_code == 200
    data = resp.json()
    assert "is_anomaly" in data
    assert "anomaly_score" in data
    assert "response_time_ms" in data
    assert data["response_time_ms"] > 0


def test_predict_empty_query_rejected(client):
    resp = client.post("/predict", json={"query": "", "use_rag": False})
    assert resp.status_code == 422


def test_predict_too_long_query_rejected(client):
    resp = client.post("/predict", json={"query": "a" * 2001, "use_rag": False})
    assert resp.status_code == 422


def test_predict_returns_scores(client):
    resp = client.post("/predict", json={"query": "Explain transformers", "use_rag": False})
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["anomaly_score"] <= 1.0
    assert 0.0 <= data["classifier_prob"] <= 1.0


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "system" in data
    assert "total_predictions" in data["system"]


def test_ingest_endpoint(client):
    resp = client.post("/ingest", json={
        "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "doc_id": "test_doc_001",
        "filename": "test.txt",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ingested"
    assert data["doc_id"] == "test_doc_001"
    assert data["chunks"] >= 1


def test_predict_with_rag_after_ingest(client):
    resp = client.post("/predict", json={"query": "What is machine learning?", "use_rag": True})
    assert resp.status_code == 200
    data = resp.json()
    assert "rag_answer" in data


def test_retrain_endpoint(client):
    resp = client.post("/retrain")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "retrained"
    assert data["auc_mean"] > 0.0
    assert data["n_features"] == 15


def test_predict_sql_injection(client):
    sql = "SELECT * FROM users; DROP TABLE predictions--"
    resp = client.post("/predict", json={"query": sql, "use_rag": False})
    assert resp.status_code == 200
    data = resp.json()
    assert "is_anomaly" in data
