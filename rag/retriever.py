import logging
from typing import Dict, List, Tuple

from rag.index import get_index

logger = logging.getLogger(__name__)

_DEFAULT_MIN_SCORE: float = 0.0


def retrieve_and_answer(
    query: str,
    top_k: int = 3,
    min_score: float = _DEFAULT_MIN_SCORE,
) -> Tuple[str, List[Dict]]:
    """Retrieve relevant chunks and synthesise an extractive answer.

    Args:
        query: User query string.
        top_k: Maximum number of context chunks to retrieve.
        min_score: Discard results whose similarity score is below this
            threshold. Defaults to 0.0 (no filtering).

    Returns:
        Tuple of (answer_text, sources) where sources is a list of dicts
        with keys doc_id, score, and excerpt.
    """
    index = get_index()

    if len(index) == 0:
        logger.info("RAG index is empty; no answer available.")
        return "No documents indexed yet.", []

    query_vec = index.embed([query])[0]
    results = index.search(query_vec, top_k=top_k)

    if min_score > _DEFAULT_MIN_SCORE:
        results = [(chunk, doc_id, score) for chunk, doc_id, score in results if score >= min_score]

    if not results:
        return "No relevant context found.", []

    context_parts: List[str] = []
    sources: List[Dict] = []
    for chunk, doc_id, score in results:
        context_parts.append(chunk)
        sources.append({"doc_id": doc_id, "score": round(score, 4), "excerpt": chunk[:120]})

    context = "\n\n---\n\n".join(context_parts)
    answer = _synthesize_answer(query, context)
    logger.debug("RAG answer synthesised from %d chunks for query len=%d", len(results), len(query))

    return answer, sources


def _score_sentences(query: str, sentences: List[str]) -> List[Tuple[float, str]]:
    """Score each sentence by token overlap with the query.

    Args:
        query: Query string used as the reference for overlap.
        sentences: Candidate sentences to score.

    Returns:
        List of (overlap_score, sentence) tuples.
    """
    q_tokens = set(query.lower().split())
    n_q = max(len(q_tokens), 1)
    scored: List[Tuple[float, str]] = [
        (len(q_tokens & set(sent.lower().split())) / n_q, sent)
        for sent in sentences
    ]
    return scored


def _synthesize_answer(query: str, context: str) -> str:
    """Return the top-3 most query-relevant sentences from context.

    Args:
        query: User query used to rank sentences.
        context: Retrieved context text (may contain multiple paragraphs).

    Returns:
        A period-joined string of up to three top-ranked sentences.
    """
    sentences = [s.strip() for s in context.replace("\n", " ").split(".") if len(s.strip()) > 20]
    if not sentences:
        return context[:500]

    scored = _score_sentences(query, sentences)
    scored.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for _, s in scored[:3]]
    return ". ".join(top_sentences) + "."
