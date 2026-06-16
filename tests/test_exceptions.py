"""Tests for the custom exception hierarchy in app/exceptions.py."""

from __future__ import annotations

import pytest

from app.exceptions import (
    DriftCheckError,
    FeatureMismatchError,
    IndexEmptyError,
    IngestError,
    ModelNotLoadedError,
    RagSentinelError,
)


def test_base_exception_is_exception():
    assert issubclass(RagSentinelError, Exception)


def test_model_not_loaded_is_base():
    assert issubclass(ModelNotLoadedError, RagSentinelError)


def test_feature_mismatch_is_both_base_and_value_error():
    assert issubclass(FeatureMismatchError, RagSentinelError)
    assert issubclass(FeatureMismatchError, ValueError)


def test_ingest_error_is_base():
    assert issubclass(IngestError, RagSentinelError)


def test_drift_check_error_is_base():
    assert issubclass(DriftCheckError, RagSentinelError)


def test_index_empty_error_is_base():
    assert issubclass(IndexEmptyError, RagSentinelError)


def test_raise_and_catch_base():
    with pytest.raises(RagSentinelError):
        raise ModelNotLoadedError("model not yet loaded")


def test_raise_and_catch_feature_mismatch_as_value_error():
    with pytest.raises(ValueError):
        raise FeatureMismatchError("expected 15 features, got 10")


@pytest.mark.parametrize(
    "exc_class",
    [ModelNotLoadedError, FeatureMismatchError, IngestError, DriftCheckError, IndexEmptyError],
)
def test_all_exceptions_catchable_as_base(exc_class):
    with pytest.raises(RagSentinelError):
        raise exc_class("test message")


def test_exception_message_preserved():
    msg = "feature count mismatch: expected 15 got 5"
    exc = FeatureMismatchError(msg)
    assert str(exc) == msg
