# Changelog

All notable changes to RAG Sentinel are documented here.

## [1.1.0] — 2026-06-16

### Added
- `/predict/batch` endpoint for scoring up to 50 queries in a single request
- `/index/stats` endpoint exposing RAG index health (chunk count, doc count, FAISS status)
- `SentinelIndex.stats()` method for introspection
- Correlation ID middleware — attaches `X-Request-ID` header to every response
- `app/middleware.py` module with `CorrelationIdMiddleware` and `RequestTimingMiddleware`
- `app/utils.py` module with `sanitize_query`, `truncate`, `clamp`, and `chunk_list` helpers
- `app/schemas.py` centralising all Pydantic request/response models
- Custom exception hierarchy in `app/exceptions.py` (6 typed exception classes)
- `compute_score_percentiles` in `app/monitoring.py` (p50/p90/p99 over anomaly scores)
- `min_score` threshold parameter in `retrieve_and_answer` to filter low-similarity results
- PEP 561 `py.typed` markers in `app/` and `rag/` packages
- `N_FEATURES` constant and `Final` type annotations for all constants
- `__all__` exports in `app` and `rag` packages
- OpenAPI tags for all endpoints (`health`, `inference`, `documents`, `monitoring`)
- PostgreSQL connection pool configuration (`DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, `DB_POOL_TIMEOUT`)
- mypy type-check and pip-audit security jobs to GitHub Actions CI
- 310+ tests across 18 modules (up from 165)

### Changed
- `Settings` derives `app_name` from `APP_NAME` constant; `LOG_LEVEL` validated at init
- `_compute_injection_features` now uses `lru_cache`-backed compiled regex patterns
- Extracted `_load_model_metrics` helper from `get_system_metrics`
- `IngestRequest.doc_id` now validates against alphanumeric + dash/dot/underscore pattern
- `retrain_log.json` rotation added (configurable via `RETRAIN_LOG_MAX_BYTES`)

### Fixed
- `int(os.getenv())` in `rag/index.py` now uses safe `_parse_int_env` with fallback logging
- Database engine creation wrapped in try/except to surface configuration errors clearly
- Query sanitisation in `/predict` strips null bytes and control characters
- `get_system_metrics` DB queries wrapped in try/except to prevent 500s on DB failures
- Feature count validated in `predict_anomaly` before scoring to catch model/feature mismatch

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
