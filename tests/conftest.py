import os
import pytest
from unittest.mock import patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_rag_sentinel.db")
os.environ.setdefault("MODEL_PATH", "/tmp/test_model.joblib")
os.environ.setdefault("METRICS_PATH", "/tmp/test_metrics.json")

from app.database import Base


@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(test_engine):
    TestingSession = sessionmaker(bind=test_engine)
    session = TestingSession()
    yield session
    session.rollback()
    session.close()


@pytest.fixture(scope="session")
def trained_model():
    from app.model import train_model
    from app.features import generate_training_corpus
    X, y = generate_training_corpus(n_normal=100, n_anomaly=20, seed=0)
    bundle, metrics = train_model(X, y)
    return bundle, metrics


@pytest.fixture(scope="session")
def api_client(trained_model):
    bundle, _ = trained_model
    with patch("app.main._model_bundle", bundle):
        from app.main import app
        from app.database import init_db
        init_db()
        client = TestClient(app, raise_server_exceptions=False)
        yield client


@pytest.fixture
def sample_features():
    from app.features import extract_query_features
    return extract_query_features("What is machine learning?", [])


@pytest.fixture
def anomalous_features():
    from app.features import extract_query_features
    return extract_query_features("DROP TABLE users; SELECT * FROM admin--", [])
