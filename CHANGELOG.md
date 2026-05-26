# Changelog

All notable changes to RAG Sentinel are documented here.

## [1.0.0] — 2026-05-26

### Added
- FastAPI app with `/predict`, `/ingest`, `/metrics`, `/retrain`, `/health`, `/version` endpoints
- RandomForest + IsolationForest ensemble anomaly detector with 15-feature pipeline
- FAISS-backed RAG index with hashed TF-IDF embedding fallback
- KS-test drift monitoring with 24-hour sliding window
- SQLAlchemy persistence layer (SQLite dev / PostgreSQL prod)
- Airflow DAG for daily drift-triggered retraining
- Request timing middleware with structured logging
- Comprehensive pytest test suite (104+ tests across 9 modules)
- Docker + docker-compose deployment configuration
- GitHub Actions CI (ruff lint + pytest)
- `.pre-commit-config.yaml`, `Makefile`, `pyproject.toml` dev toolchain
- Database indexes on `created_at` and `is_anomaly` for query performance
- `pool_pre_ping=True` for resilient database connections

### Changed
- Replaced deprecated `datetime.utcnow()` with timezone-aware `datetime.now(timezone.utc)`
- Extracted `_compute_rolling_stats` and `_compute_injection_features` helpers in `features.py`
- Extracted `_score_sentences` helper in `retriever.py`
- Extracted `_get_rag_context` helper in `main.py`
- Added `__repr__` to all ORM models

### Fixed
- Metrics file read now wrapped in try/except to handle invalid JSON gracefully
- FAISS operations wrapped in try/except with NumPy fallback
- Empty document ingestion handled gracefully (returns 0 chunks)
