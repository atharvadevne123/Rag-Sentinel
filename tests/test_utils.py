"""Tests for app/utils.py — sanitize_query, truncate, clamp, chunk_list."""

from __future__ import annotations

import pytest

from app.utils import chunk_list, clamp, sanitize_query, truncate

# ---------------------------------------------------------------------------
# sanitize_query
# ---------------------------------------------------------------------------


def test_sanitize_removes_null_byte():
    assert sanitize_query("hello\x00world") == "helloworld"


def test_sanitize_removes_ascii_control():
    assert sanitize_query("a\x01b\x1fc") == "abc"


def test_sanitize_preserves_tab():
    assert "\t" in sanitize_query("col1\tcol2")


def test_sanitize_preserves_newline():
    assert "\n" in sanitize_query("line1\nline2")


def test_sanitize_preserves_normal_text():
    text = "What is machine learning?"
    assert sanitize_query(text) == text


def test_sanitize_removes_del():
    assert sanitize_query("a\x7fb") == "ab"


def test_sanitize_empty_string():
    assert sanitize_query("") == ""


def test_sanitize_all_control_chars():
    dirty = "".join(chr(i) for i in range(32)) + "clean"
    result = sanitize_query(dirty)
    assert result.endswith("clean")
    assert "\t" in result
    assert "\n" in result


# ---------------------------------------------------------------------------
# truncate
# ---------------------------------------------------------------------------


def test_truncate_short_string_unchanged():
    assert truncate("hello", 10) == "hello"


def test_truncate_exactly_max_unchanged():
    assert truncate("hello", 5) == "hello"


def test_truncate_longer_string():
    result = truncate("hello world", 8)
    assert len(result) <= 8
    assert result.endswith("…")


def test_truncate_custom_suffix():
    result = truncate("abcdefgh", 5, suffix="...")
    assert result.endswith("...")
    assert len(result) == 5


def test_truncate_zero_max():
    result = truncate("hello", 0)
    assert result == "…"


def test_truncate_empty_string():
    assert truncate("", 10) == ""


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------


def test_clamp_within_range_unchanged():
    assert clamp(0.5, 0.0, 1.0) == 0.5


def test_clamp_below_lo_returns_lo():
    assert clamp(-1.0, 0.0, 1.0) == 0.0


def test_clamp_above_hi_returns_hi():
    assert clamp(2.0, 0.0, 1.0) == 1.0


def test_clamp_at_lo_boundary():
    assert clamp(0.0, 0.0, 1.0) == 0.0


def test_clamp_at_hi_boundary():
    assert clamp(1.0, 0.0, 1.0) == 1.0


def test_clamp_negative_range():
    assert clamp(-5.0, -10.0, -1.0) == -5.0


# ---------------------------------------------------------------------------
# chunk_list
# ---------------------------------------------------------------------------


def test_chunk_list_basic():
    result = chunk_list([1, 2, 3, 4, 5], 2)
    assert result == [[1, 2], [3, 4], [5]]


def test_chunk_list_exact_multiple():
    result = chunk_list([1, 2, 3, 4], 2)
    assert result == [[1, 2], [3, 4]]


def test_chunk_list_size_one():
    result = chunk_list([1, 2, 3], 1)
    assert result == [[1], [2], [3]]


def test_chunk_list_size_larger_than_list():
    result = chunk_list([1, 2], 10)
    assert result == [[1, 2]]


def test_chunk_list_empty():
    assert chunk_list([], 3) == []


def test_chunk_list_size_zero_raises():
    with pytest.raises(ValueError, match="size must be >= 1"):
        chunk_list([1, 2, 3], 0)


def test_chunk_list_negative_size_raises():
    with pytest.raises(ValueError):
        chunk_list([1, 2], -1)


def test_chunk_list_all_elements_preserved():
    items = list(range(17))
    chunks = chunk_list(items, 5)
    flat = [x for c in chunks for x in c]
    assert flat == items
