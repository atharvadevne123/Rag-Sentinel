"""RAG sub-package: document ingestion, vector indexing, and retrieval."""

from rag.index import SentinelIndex, get_index, reset_index
from rag.ingest import chunk_text, clean_text, ingest_document
from rag.retriever import retrieve_and_answer

__all__ = [
    "SentinelIndex",
    "get_index",
    "reset_index",
    "chunk_text",
    "clean_text",
    "ingest_document",
    "retrieve_and_answer",
]
