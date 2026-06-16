import hashlib
import logging
import os
from typing import List, Optional, Tuple

import numpy as np

def _parse_int_env(name: str, default: int, minimum: int = 1) -> int:
    """Parse an integer environment variable with a safe fallback on error."""
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
        if value < minimum:
            raise ValueError(f"{name}={value} is below minimum {minimum}")
        return value
    except (ValueError, TypeError):
        logging.getLogger(__name__).warning(
            "Invalid value %r for env var %s; using default %d", raw, name, default
        )
        return default


CHUNK_SIZE = _parse_int_env("CHUNK_SIZE", 128, minimum=1)
CHUNK_OVERLAP = _parse_int_env("CHUNK_OVERLAP", 16, minimum=0)
EMBED_DIM = 384
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "rag_index.faiss")

logger = logging.getLogger(__name__)

_index_instance: Optional["SentinelIndex"] = None


class SentinelIndex:
    """FAISS-backed vector index with a hashed TF-IDF embedding fallback.

    When faiss-cpu is installed the index uses inner-product search over
    L2-normalised embeddings (cosine similarity). Without FAISS, a NumPy
    cosine-similarity scan is used instead.
    """

    def __init__(self) -> None:
        self.vectors: List[np.ndarray] = []
        self.chunks: List[str] = []
        self.doc_ids: List[str] = []
        self._faiss_available: bool = False
        self._faiss_index = None
        self._try_init_faiss()

    def _try_init_faiss(self) -> None:
        """Attempt to initialise a FAISS IndexFlatIP; fall back silently."""
        try:
            import faiss  # type: ignore[import]

            self._faiss_index = faiss.IndexFlatIP(EMBED_DIM)
            self._faiss_available = True
            logger.debug("FAISS index initialised (dim=%d)", EMBED_DIM)
        except ImportError:
            logger.info("faiss-cpu not installed; using NumPy cosine-similarity fallback")
            self._faiss_available = False

    def embed(self, texts: List[str]) -> np.ndarray:
        """Produce fixed-dimension embeddings via hashed TF-IDF projection.

        Args:
            texts: List of text strings to embed.

        Returns:
            Float32 array of shape (len(texts), EMBED_DIM), L2-normalised.
        """
        result: List[np.ndarray] = []
        for text in texts:
            tokens = text.lower().split()
            vec = np.zeros(EMBED_DIM, dtype=np.float32)
            for token in tokens:
                h = int(hashlib.md5(token.encode()).hexdigest(), 16) % EMBED_DIM
                vec[h] += 1.0
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            result.append(vec)
        return np.array(result, dtype=np.float32)

    def add(self, embeddings: np.ndarray, chunks: List[str], doc_id: str) -> None:
        """Add embedded chunks to the index.

        Args:
            embeddings: Array of shape (n_chunks, EMBED_DIM).
            chunks: Corresponding text chunk strings.
            doc_id: Identifier for the source document.
        """
        for emb, chunk in zip(embeddings, chunks):
            self.vectors.append(emb)
            self.chunks.append(chunk)
            self.doc_ids.append(doc_id)

        if self._faiss_available and self._faiss_index is not None:
            try:
                self._faiss_index.add(embeddings)
                logger.debug("Added %d chunks for doc_id=%s to FAISS", len(chunks), doc_id)
            except Exception as exc:
                logger.error("FAISS add failed for doc_id=%s: %s", doc_id, exc)

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Retrieve the top-k most similar chunks for a query vector.

        Args:
            query_vec: Embedding of shape (EMBED_DIM,).
            top_k: Number of results to return.

        Returns:
            List of (chunk_text, doc_id, score) tuples, highest score first.
        """
        if not self.vectors:
            return []

        if self._faiss_available and self._faiss_index is not None and self._faiss_index.ntotal > 0:
            try:
                q = query_vec.reshape(1, -1)
                scores, indices = self._faiss_index.search(q, min(top_k, len(self.chunks)))
                results: List[Tuple[str, str, float]] = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0:
                        results.append((self.chunks[idx], self.doc_ids[idx], float(score)))
                return results
            except Exception as exc:
                logger.warning("FAISS search failed, falling back to NumPy: %s", exc)

        # NumPy cosine similarity fallback
        matrix = np.array(self.vectors)
        sims = matrix @ query_vec
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[i], self.doc_ids[i], float(sims[i])) for i in top_indices]

    def stats(self) -> dict:
        """Return a summary of the index state.

        Returns:
            Dict with chunk_count, doc_count, faiss_available, and faiss_total keys.
        """
        return {
            "chunk_count": len(self.chunks),
            "doc_count": len(set(self.doc_ids)),
            "faiss_available": self._faiss_available,
            "faiss_total": self._faiss_index.ntotal if self._faiss_available and self._faiss_index else 0,
        }

    def __len__(self) -> int:
        return len(self.chunks)

    def __repr__(self) -> str:
        return f"<SentinelIndex chunks={len(self.chunks)} faiss={self._faiss_available}>"


def get_index() -> SentinelIndex:
    """Return the module-level singleton SentinelIndex, creating it if needed."""
    global _index_instance
    if _index_instance is None:
        _index_instance = SentinelIndex()
    return _index_instance


def reset_index() -> None:
    """Replace the singleton index with a fresh, empty instance."""
    global _index_instance
    _index_instance = SentinelIndex()
