import re
from typing import List

from rag.index import CHUNK_OVERLAP, CHUNK_SIZE, get_index

_chunk_store: dict = {}  # doc_id -> list of chunk texts


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return chunks


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def ingest_document(text: str, doc_id: str) -> int:
    text = clean_text(text)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    index = get_index()
    embeddings = index.embed(chunks)
    index.add(embeddings, chunks, doc_id)

    _chunk_store[doc_id] = chunks
    return len(chunks)


def get_stored_chunks(doc_id: str) -> List[str]:
    return _chunk_store.get(doc_id, [])
