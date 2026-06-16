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
    resp = client.post(
        "/ingest",
        json={
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "doc_id": "test_doc_001",
            "filename": "test.txt",
        },
    )
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


def test_version_endpoint(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    data = resp.json()
    assert "version" in data
    assert data["version"] == "1.0.0"


def test_predict_top_k_min(client):
    resp = client.post("/predict", json={"query": "Explain transformers", "use_rag": False, "top_k": 1})
    assert resp.status_code == 200


def test_predict_top_k_max(client):
    resp = client.post("/predict", json={"query": "Explain transformers", "use_rag": False, "top_k": 10})
    assert resp.status_code == 200


def test_predict_top_k_invalid_rejected(client):
    resp = client.post("/predict", json={"query": "test", "use_rag": False, "top_k": 0})
    assert resp.status_code == 422


def test_ingest_short_text_rejected(client):
    resp = client.post("/ingest", json={"text": "hi", "doc_id": "x"})
    assert resp.status_code == 422


def test_health_model_loaded(client):
    resp = client.get("/health")
    assert resp.json()["model_loaded"] is True


def test_health_includes_database_status(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "database_ok" in data
    assert isinstance(data["database_ok"], bool)


def test_predict_response_has_all_fields(client):
    resp = client.post("/predict", json={"query": "What is deep learning?", "use_rag": False})
    data = resp.json()
    required = {"query", "is_anomaly", "anomaly_score", "classifier_prob", "isolation_score", "response_time_ms"}
    assert required.issubset(data.keys())


def test_retrain_returns_n_features_15(client):
    resp = client.post("/retrain")
    data = resp.json()
    assert data["n_features"] == 15


def test_ingest_returns_ingested_status(client):
    resp = client.post(
        "/ingest",
        json={
            "text": "Transformer models revolutionised NLP with attention mechanisms. " * 5,
            "doc_id": "transformer_doc",
        },
    )
    assert resp.json()["status"] == "ingested"


def test_index_stats_endpoint(client):
    resp = client.get("/index/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "chunk_count" in data
    assert "doc_count" in data
    assert "faiss_available" in data


def test_predict_response_has_x_request_id_header(client):
    resp = client.post("/predict", json={"query": "What is AI?", "use_rag": False})
    assert "x-request-id" in resp.headers


def test_predict_custom_request_id_echoed(client):
    custom_id = "test-correlation-id-123"
    resp = client.post(
        "/predict",
        json={"query": "What is AI?", "use_rag": False},
        headers={"X-Request-ID": custom_id},
    )
    assert resp.headers.get("x-request-id") == custom_id


@pytest.mark.parametrize(
    "query",
    [
        "What is supervised learning?",
        "Explain gradient descent",
        "How do neural networks work?",
        "What is the bias-variance tradeoff?",
    ],
)
def test_predict_normal_queries_parametrized(client, query):
    resp = client.post("/predict", json={"query": query, "use_rag": False})
    assert resp.status_code == 200
    data = resp.json()
    assert data["response_time_ms"] > 0


def test_ingest_invalid_doc_id_rejected(client):
    resp = client.post(
        "/ingest",
        json={
            "text": "Machine learning is a field of AI. " * 5,
            "doc_id": "doc id with spaces",
        },
    )
    assert resp.status_code == 422


def test_metrics_has_drift_key(client):
    resp = client.get("/metrics")
    data = resp.json()
    assert "drift" in data


def test_predict_query_returned_in_response(client):
    query = "What is reinforcement learning?"
    resp = client.post("/predict", json={"query": query, "use_rag": False})
    data = resp.json()
    assert data["query"] == query
