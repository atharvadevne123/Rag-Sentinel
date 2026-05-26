"""Centralised application configuration loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache


class Settings:
    """Application settings resolved from environment variables at startup.

    All values fall back to sensible development defaults so the app works
    out-of-the-box without a .env file.
    """

    database_url: str
    model_path: str
    metrics_path: str
    faiss_index_path: str
    chunk_size: int
    chunk_overlap: int
    log_level: str
    app_version: str

    def __init__(self) -> None:
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./rag_sentinel.db")
        self.model_path = os.getenv("MODEL_PATH", "model.joblib")
        self.metrics_path = os.getenv("METRICS_PATH", "metrics.json")
        self.faiss_index_path = os.getenv("FAISS_INDEX_PATH", "rag_index.faiss")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "128"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "16"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.app_version = os.getenv("APP_VERSION", "1.0.0")

    def __repr__(self) -> str:
        return (
            f"Settings(database_url={self.database_url!r}, "
            f"model_path={self.model_path!r}, "
            f"log_level={self.log_level!r})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()
