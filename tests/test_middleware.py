"""Tests for app/middleware.py — CorrelationIdMiddleware and RequestTimingMiddleware."""

from __future__ import annotations

import re

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import CorrelationIdMiddleware, RequestTimingMiddleware


def _make_app(*middleware_classes) -> FastAPI:
    """Build a minimal FastAPI app wired with the given middleware classes."""
    mini = FastAPI()
    for cls in middleware_classes:
        mini.add_middleware(cls)

    @mini.get("/ping")
    def ping():
        return {"pong": True}

    return mini


# ---------------------------------------------------------------------------
# CorrelationIdMiddleware
# ---------------------------------------------------------------------------


def test_correlation_id_generated_when_absent():
    client = TestClient(_make_app(CorrelationIdMiddleware))
    resp = client.get("/ping")
    assert "x-request-id" in resp.headers
    assert len(resp.headers["x-request-id"]) > 0


def test_correlation_id_is_uuid_format_when_generated():
    client = TestClient(_make_app(CorrelationIdMiddleware))
    resp = client.get("/ping")
    rid = resp.headers["x-request-id"]
    uuid_pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    assert uuid_pattern.match(rid), f"Not a UUID: {rid}"


def test_correlation_id_echoed_when_provided():
    client = TestClient(_make_app(CorrelationIdMiddleware))
    custom_id = "my-trace-id-xyz"
    resp = client.get("/ping", headers={"X-Request-ID": custom_id})
    assert resp.headers["x-request-id"] == custom_id


def test_correlation_id_unique_across_requests():
    client = TestClient(_make_app(CorrelationIdMiddleware))
    ids = {client.get("/ping").headers["x-request-id"] for _ in range(5)}
    assert len(ids) == 5, "Expected all 5 request IDs to be unique"


def test_correlation_id_present_on_non_200_response():
    mini = FastAPI()
    mini.add_middleware(CorrelationIdMiddleware)

    @mini.get("/err")
    def err():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="not found")

    client = TestClient(mini, raise_server_exceptions=False)
    resp = client.get("/err")
    assert "x-request-id" in resp.headers


# ---------------------------------------------------------------------------
# RequestTimingMiddleware
# ---------------------------------------------------------------------------


def test_request_timing_does_not_drop_response(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="app.middleware"):
        client = TestClient(_make_app(RequestTimingMiddleware))
        resp = client.get("/ping")
    assert resp.status_code == 200


def test_request_timing_logs_method_and_path(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="app.middleware"):
        client = TestClient(_make_app(RequestTimingMiddleware))
        client.get("/ping")
    assert any("GET" in r.message and "/ping" in r.message for r in caplog.records)


def test_request_timing_logs_status_code(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="app.middleware"):
        client = TestClient(_make_app(RequestTimingMiddleware))
        client.get("/ping")
    assert any("200" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Both middleware stacked
# ---------------------------------------------------------------------------


def test_both_middleware_stack_without_conflict():
    client = TestClient(_make_app(CorrelationIdMiddleware, RequestTimingMiddleware))
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert "x-request-id" in resp.headers
