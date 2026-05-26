from __future__ import annotations

import os

import pytest


def test_settings_defaults():
    from app.config import Settings
    s = Settings()
    assert "rag_sentinel" in s.database_url or "sqlite" in s.database_url
    assert s.model_path.endswith(".joblib") or s.model_path == os.getenv("MODEL_PATH", "model.joblib")
    assert s.chunk_size > 0
    assert s.chunk_overlap >= 0
    assert s.log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def test_settings_env_override(monkeypatch):
    from app.config import Settings
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/test_db")
    monkeypatch.setenv("CHUNK_SIZE", "64")
    s = Settings()
    assert s.database_url == "postgresql://user:pass@localhost/test_db"
    assert s.chunk_size == 64


def test_settings_repr():
    from app.config import Settings
    s = Settings()
    r = repr(s)
    assert "Settings(" in r
    assert "database_url=" in r


def test_get_settings_is_cached():
    from app.config import get_settings
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_settings_chunk_overlap_default():
    from app.config import Settings
    s = Settings()
    assert s.chunk_overlap == int(os.getenv("CHUNK_OVERLAP", "16"))
