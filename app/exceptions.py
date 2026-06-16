"""Custom exception classes for RAG Sentinel.

Using specific exception types makes error handling more precise and
allows callers to catch only the errors they can actually recover from.
"""

from __future__ import annotations


class RagSentinelError(Exception):
    """Base exception for all RAG Sentinel application errors."""


class ModelNotLoadedError(RagSentinelError):
    """Raised when a prediction is attempted before the model is loaded."""


class FeatureMismatchError(RagSentinelError, ValueError):
    """Raised when the feature vector size does not match the trained model."""


class IngestError(RagSentinelError):
    """Raised when document ingestion fails due to index or chunking errors."""


class DriftCheckError(RagSentinelError):
    """Raised when a drift-detection query cannot complete."""


class IndexEmptyError(RagSentinelError):
    """Raised when a retrieval is attempted against an empty RAG index."""
