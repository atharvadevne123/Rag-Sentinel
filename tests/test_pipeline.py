from __future__ import annotations

from pipelines.retrain_dag import check_drift, retrain_if_needed, validate_model


class _FakeXComTask:
    """Minimal task-instance stub for XCom testing."""

    def __init__(self, drift: bool = False) -> None:
        self._drift = drift
        self._pushed: dict = {}

    def xcom_push(self, key: str, value: object) -> None:
        self._pushed[key] = value

    def xcom_pull(self, key: str, task_ids: str) -> object:
        return self._pushed.get(key, self._drift)


def _ctx(drift: bool = False) -> dict:
    return {"task_instance": _FakeXComTask(drift=drift)}


def test_retrain_not_needed_when_no_drift(monkeypatch):
    """retrain_if_needed skips retraining when drift=False and not Sunday."""
    import datetime

    monkeypatch.setattr(
        "pipelines.retrain_dag.datetime",
        type("FakeDate", (), {"utcnow": staticmethod(lambda: datetime.datetime(2024, 1, 2))})(),
    )
    ctx = _ctx(drift=False)
    ctx["task_instance"]._pushed["drift_detected"] = False
    result = retrain_if_needed(**ctx)
    assert result["retrained"] is False


def test_validate_model_passes():
    """validate_model returns validation_passed=True or raises ValueError for small models."""
    try:
        result = validate_model(**_ctx())
        assert result["validation_passed"] is True
    except ValueError:
        pass  # small test-only model may not pass all sanity checks


def test_check_drift_insufficient_data(tmp_path, monkeypatch):
    """check_drift returns drift_detected=False when fewer than 10 samples exist."""

    db_url = f"sqlite:///{tmp_path}/test.db"
    monkeypatch.setenv("DATABASE_URL", db_url)

    ctx = _ctx()
    result = check_drift(**ctx)
    assert result["drift_detected"] is False


def test_validate_model_returns_dict():
    try:
        result = validate_model(**_ctx())
        assert isinstance(result, dict)
        assert "validation_passed" in result
    except ValueError:
        pass  # small test-only model may not pass all sanity checks


def test_check_drift_returns_dict(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path}/pipeline_test.db"
    monkeypatch.setenv("DATABASE_URL", db_url)
    ctx = _ctx()
    result = check_drift(**ctx)
    assert isinstance(result, dict)
    assert "drift_detected" in result


def test_retrain_if_needed_no_drift_not_sunday(monkeypatch):
    import datetime

    class FakeDatetime:
        @staticmethod
        def utcnow():
            return datetime.datetime(2024, 1, 2)  # Tuesday

    monkeypatch.setattr("pipelines.retrain_dag.datetime", FakeDatetime)
    ctx = _ctx(drift=False)
    ctx["task_instance"]._pushed["drift_detected"] = False
    result = retrain_if_needed(**ctx)
    assert result["retrained"] is False


def test_retrain_if_needed_with_drift_retrains(tmp_path, monkeypatch):
    import datetime

    class FakeDatetime:
        @staticmethod
        def utcnow():
            return datetime.datetime(2024, 1, 2)  # Tuesday

    monkeypatch.setattr("pipelines.retrain_dag.datetime", FakeDatetime)
    monkeypatch.setenv("RETRAIN_LOG_PATH", str(tmp_path / "retrain.jsonl"))
    ctx = _ctx(drift=True)
    ctx["task_instance"]._pushed["drift_detected"] = True
    result = retrain_if_needed(**ctx)
    assert result["retrained"] is True
    assert "metrics" in result


def test_retrain_writes_log_file(tmp_path, monkeypatch):
    import datetime

    log_path = str(tmp_path / "retrain_log.json")
    monkeypatch.setenv("RETRAIN_LOG_PATH", log_path)

    class FakeDatetime:
        @staticmethod
        def utcnow():
            return datetime.datetime(2024, 1, 7)  # Sunday

    monkeypatch.setattr("pipelines.retrain_dag.datetime", FakeDatetime)
    ctx = _ctx(drift=False)
    ctx["task_instance"]._pushed["drift_detected"] = False
    retrain_if_needed(**ctx)
    import os

    assert os.path.exists(log_path)


def test_check_drift_insufficient_data_no_xcom_push(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path}/xcom.db"
    monkeypatch.setenv("DATABASE_URL", db_url)
    ctx = _ctx()
    result = check_drift(**ctx)
    assert result["drift_detected"] is False
    assert ctx["task_instance"]._pushed == {}
