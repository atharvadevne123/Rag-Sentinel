.PHONY: install test test-cov lint format type-check run docker-build docker-up docker-down clean audit

install:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio httpx pytest-cov ruff mypy

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short \
	  --cov=app --cov=rag --cov=pipelines \
	  --cov-report=term-missing \
	  --cov-report=html:htmlcov

lint:
	ruff check . --select E,F,W,I --ignore E501
	ruff format --check .

format:
	ruff check . --select E,F,W,I --ignore E501 --fix
	ruff format .

type-check:
	mypy app/ rag/ --ignore-missing-imports --no-error-summary

audit:
	pip install pip-audit -q && pip-audit -r requirements.txt --skip-editable

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t rag-sentinel:latest .

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	rm -f model.joblib metrics.json rag_sentinel.db test_rag_sentinel.db coverage.xml
	rm -rf htmlcov/ .mypy_cache/ .ruff_cache/ dist/ build/ *.egg-info/
