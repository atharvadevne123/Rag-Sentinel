"""General-purpose utility helpers for RAG Sentinel.

Keep this module free of cross-app imports so it can be used anywhere
without creating circular dependencies.
"""

from __future__ import annotations

import re
from typing import List

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_query(text: str) -> str:
    """Strip null bytes and ASCII control characters from user input.

    Tabs (\\t) and newlines (\\n) are preserved; all other control
    characters below U+0020 and DEL (U+007F) are removed.

    Args:
        text: Raw input string.

    Returns:
        Cleaned string safe for feature extraction and logging.
    """
    return _CONTROL_CHAR_RE.sub("", text)


def truncate(text: str, max_chars: int, suffix: str = "…") -> str:
    """Truncate *text* to at most *max_chars* characters.

    If truncation occurs the suffix (default ``…``) is appended so callers
    can tell the string was shortened.

    Args:
        text: Input string.
        max_chars: Maximum character count including the suffix.
        suffix: String appended when truncation happens.

    Returns:
        Original string if short enough, otherwise a truncated string ending
        with *suffix*.
    """
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - len(suffix))
    return text[:keep] + suffix


def clamp(value: float, lo: float, hi: float) -> float:
    """Return *value* clamped to the closed interval [*lo*, *hi*].

    Args:
        value: The value to clamp.
        lo: Lower bound (inclusive).
        hi: Upper bound (inclusive).

    Returns:
        Clamped float.
    """
    return max(lo, min(value, hi))


def chunk_list(items: list, size: int) -> List[list]:
    """Partition *items* into contiguous sublists of at most *size* elements.

    Args:
        items: Input list.
        size: Maximum chunk size (must be >= 1).

    Returns:
        List of sublists; the last sublist may be shorter than *size*.

    Raises:
        ValueError: If *size* < 1.
    """
    if size < 1:
        raise ValueError(f"chunk_list size must be >= 1, got {size}")
    return [items[i : i + size] for i in range(0, len(items), size)]
