"""Centralised application configuration loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache

from app.constants import APP_NAME
from app.constants import APP_VERSION as _DEFAULT_VERSION

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class Settings:
    """Application settings resolved from environment variables at startup.

    All values fall back to sensible development defaults so the app works
    out-of-the-box without a .env file.
    """

    app_name: str
    database_url: str
    model_path: str
    metrics_path: str
    faiss_index_path: str
    chunk_size: int
    chunk_overlap: int
    log_level: str
    app_version: str

    def __init__(self) -> None:
        self.app_name = APP_NAME
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./rag_sentinel.db")
        self.model_path = os.getenv("MODEL_PATH", "model.joblib")
        self.metrics_path = os.getenv("METRICS_PATH", "metrics.json")
        self.faiss_index_path = os.getenv("FAISS_INDEX_PATH", "rag_index.faiss")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "128"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "16"))
        raw_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_level = raw_level if raw_level in _VALID_LOG_LEVELS else "INFO"
        self.app_version = os.getenv("APP_VERSION", _DEFAULT_VERSION)

    def __repr__(self) -> str:
        return (
            f"Settings(app_name={self.app_name!r}, "
            f"database_url={self.database_url!r}, "
            f"model_path={self.model_path!r}, "
            f"log_level={self.log_level!r})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()
