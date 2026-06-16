from __future__ import annotations

import os


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


def test_settings_log_level_override(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    s = Settings()
    assert s.log_level == "DEBUG"


def test_settings_model_path_override(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("MODEL_PATH", "/custom/path/model.pkl")
    s = Settings()
    assert s.model_path == "/custom/path/model.pkl"


def test_settings_faiss_index_path_default():
    from app.config import Settings

    s = Settings()
    assert "faiss" in s.faiss_index_path.lower() or s.faiss_index_path.endswith(".faiss")


def test_settings_app_version_default():
    from app.config import Settings

    s = Settings()
    assert s.app_version == os.getenv("APP_VERSION", "1.0.0")


def test_settings_chunk_size_type():
    from app.config import Settings

    s = Settings()
    assert isinstance(s.chunk_size, int)
    assert isinstance(s.chunk_overlap, int)


def test_settings_all_attributes_present():
    from app.config import Settings

    s = Settings()
    required = ["database_url", "model_path", "metrics_path", "faiss_index_path",
                "chunk_size", "chunk_overlap", "log_level", "app_version"]
    for attr in required:
        assert hasattr(s, attr), f"Settings missing attribute: {attr}"
