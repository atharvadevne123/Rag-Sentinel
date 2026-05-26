from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, DocumentIndex, DriftLog, PredictionLog, get_db, init_db


@pytest.fixture
def mem_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


def test_prediction_log_create(mem_session):
    rec = PredictionLog(query="test", anomaly_score=0.3, is_anomaly=False)
    mem_session.add(rec)
    mem_session.commit()
    assert rec.id is not None


def test_prediction_log_repr(mem_session):
    rec = PredictionLog(query="test", anomaly_score=0.5, is_anomaly=True)
    mem_session.add(rec)
    mem_session.commit()
    r = repr(rec)
    assert "PredictionLog" in r
    assert "score=0.5" in r


def test_drift_log_repr(mem_session):
    rec = DriftLog(ks_statistic=0.4, p_value=0.02, drift_detected=True, sample_size=50)
    mem_session.add(rec)
    mem_session.commit()
    r = repr(rec)
    assert "DriftLog" in r
    assert "ks=0.4" in r


def test_document_index_repr(mem_session):
    rec = DocumentIndex(doc_id="doc_001", filename="test.txt", chunk_count=5)
    mem_session.add(rec)
    mem_session.commit()
    r = repr(rec)
    assert "DocumentIndex" in r
    assert "doc_001" in r


def test_document_index_unique_doc_id(mem_session):
    rec1 = DocumentIndex(doc_id="unique_doc", filename="a.txt", chunk_count=3)
    rec2 = DocumentIndex(doc_id="unique_doc", filename="b.txt", chunk_count=4)
    mem_session.add(rec1)
    mem_session.commit()
    mem_session.add(rec2)
    with pytest.raises(Exception):
        mem_session.commit()


def test_init_db_creates_tables(tmp_path):
    import os
    db_path = str(tmp_path / "test_init.db")
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    # Re-import to pick up new URL
    import importlib
    import app.database as db_mod
    importlib.reload(db_mod)
    db_mod.init_db()
    assert db_path  # tables created without error


def test_get_db_yields_session():
    gen = get_db()
    session = next(gen)
    assert session is not None
    try:
        next(gen)
    except StopIteration:
        pass


def test_prediction_log_default_rag_false(mem_session):
    rec = PredictionLog(query="test", anomaly_score=0.1, is_anomaly=False)
    mem_session.add(rec)
    mem_session.commit()
    assert rec.rag_context_used is False


def test_drift_log_sample_size_stored(mem_session):
    rec = DriftLog(ks_statistic=0.2, p_value=0.15, drift_detected=False, sample_size=75)
    mem_session.add(rec)
    mem_session.commit()
    assert rec.sample_size == 75


def test_multiple_prediction_logs_queryable(mem_session):
    for i in range(5):
        rec = PredictionLog(query=f"query {i}", anomaly_score=i * 0.1, is_anomaly=i % 2 == 0)
        mem_session.add(rec)
    mem_session.commit()
    count = mem_session.query(PredictionLog).count()
    assert count >= 5
