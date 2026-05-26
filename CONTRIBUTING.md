# Contributing to RAG Sentinel

## Getting Started

```bash
git clone https://github.com/atharvadevne123/Rag-Sentinel.git
cd Rag-Sentinel
pip install -r requirements.txt
pip install pytest ruff pre-commit
pre-commit install
```

## Development Workflow

1. Create a feature branch: `git checkout -b feat/your-feature`
2. Make your changes with atomic commits
3. Run linting: `make lint`
4. Run tests: `make test`
5. Push and open a pull request against `main`

## Code Style

- Python 3.11+, type annotations on all public functions
- `ruff` for linting and formatting (configured in `pyproject.toml`)
- Google-style docstrings on all public classes and functions
- Use `logging.getLogger(__name__)` — no bare `print()` calls

## Testing

- All new features must include pytest tests
- Tests live in `tests/` and follow the `test_<module>.py` convention
- Aim for 80%+ coverage on new code

## Pull Request Checklist

- [ ] `make lint` passes
- [ ] `make test` passes
- [ ] New/updated functions have type annotations and docstrings
- [ ] PR description explains the *why*, not just the *what*
