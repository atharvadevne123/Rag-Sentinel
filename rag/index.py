import os
import hashlib
import numpy as np
from typing import List, Tuple

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "128"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "16"))
EMBED_DIM = 384
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "rag_index.faiss")

_index_instance = None


class SentinelIndex:
    """Lightweight FAISS-backed index with TF-IDF-style embedding fallback."""

    def __init__(self):
        self.vectors: List[np.ndarray] = []
        self.chunks: List[str] = []
        self.doc_ids: List[str] = []
        self._vocab: dict = {}
        self._idf: dict = {}
        self._faiss_index = None
        self._try_init_faiss()

    def _try_init_faiss(self):
        try:
            import faiss
            self._faiss_index = faiss.IndexFlatIP(EMBED_DIM)
            self._faiss_available = True
        except ImportError:
            self._faiss_available = False

    def embed(self, texts: List[str]) -> np.ndarray:
        """Produce fixed-dim embeddings via hashed TF-IDF projection."""
        result = []
        for text in texts:
            tokens = text.lower().split()
            vec = np.zeros(EMBED_DIM, dtype=np.float32)
            for token in tokens:
                h = int(hashlib.md5(token.encode()).hexdigest(), 16) % EMBED_DIM
                vec[h] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            result.append(vec)
        return np.array(result, dtype=np.float32)

    def add(self, embeddings: np.ndarray, chunks: List[str], doc_id: str):
        for i, (emb, chunk) in enumerate(zip(embeddings, chunks)):
            self.vectors.append(emb)
            self.chunks.append(chunk)
            self.doc_ids.append(doc_id)

        if self._faiss_available and self._faiss_index is not None:
            self._faiss_index.add(embeddings)

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> List[Tuple[str, str, float]]:
        if not self.vectors:
            return []

        if self._faiss_available and self._faiss_index is not None and self._faiss_index.ntotal > 0:
            q = query_vec.reshape(1, -1)
            scores, indices = self._faiss_index.search(q, min(top_k, len(self.chunks)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    results.append((self.chunks[idx], self.doc_ids[idx], float(score)))
            return results

        # Numpy cosine similarity fallback
        matrix = np.array(self.vectors)
        sims = matrix @ query_vec
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[i], self.doc_ids[i], float(sims[i])) for i in top_indices]

    def __len__(self):
        return len(self.chunks)


def get_index() -> SentinelIndex:
    global _index_instance
    if _index_instance is None:
        _index_instance = SentinelIndex()
    return _index_instance


def reset_index():
    global _index_instance
    _index_instance = SentinelIndex()
