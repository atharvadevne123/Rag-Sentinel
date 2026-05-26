import logging
import re
from typing import Dict, List

from rag.index import CHUNK_OVERLAP, CHUNK_SIZE, get_index

logger = logging.getLogger(__name__)

_chunk_store: Dict[str, List[str]] = {}


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping word-level chunks.

    Args:
        text: Whitespace-tokenised input text.
        size: Maximum number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return chunks


def clean_text(text: str) -> str:
    """Normalise whitespace and strip non-ASCII characters.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text string.
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def ingest_document(text: str, doc_id: str) -> int:
    """Clean, chunk, embed, and index a document.

    Args:
        text: Raw document text.
        doc_id: Unique identifier for this document.

    Returns:
        Number of chunks indexed (0 if the document was empty after cleaning).
    """
    text = clean_text(text)
    chunks = chunk_text(text)
    if not chunks:
        logger.warning("Document %s produced no chunks after cleaning; skipping.", doc_id)
        return 0

    index = get_index()
    embeddings = index.embed(chunks)
    index.add(embeddings, chunks, doc_id)

    _chunk_store[doc_id] = chunks
    logger.info("Ingested doc_id=%s: %d chunks indexed.", doc_id, len(chunks))
    return len(chunks)


def get_stored_chunks(doc_id: str) -> List[str]:
    """Return the in-memory chunk list for a previously ingested document.

    Args:
        doc_id: Document identifier.

    Returns:
        List of chunk strings, or an empty list if the document is unknown.
    """
    return _chunk_store.get(doc_id, [])
