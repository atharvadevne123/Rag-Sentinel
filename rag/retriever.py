from typing import List, Tuple

from rag.index import get_index


def retrieve_and_answer(query: str, top_k: int = 3) -> Tuple[str, List[dict]]:
    index = get_index()

    if len(index) == 0:
        return "No documents indexed yet.", []

    query_vec = index.embed([query])[0]
    results = index.search(query_vec, top_k=top_k)

    if not results:
        return "No relevant context found.", []

    context_parts = []
    sources = []
    for chunk, doc_id, score in results:
        context_parts.append(chunk)
        sources.append({"doc_id": doc_id, "score": round(score, 4), "excerpt": chunk[:120]})

    context = "\n\n---\n\n".join(context_parts)
    answer = _synthesize_answer(query, context)

    return answer, sources


def _synthesize_answer(query: str, context: str) -> str:
    """Extractive summarization: returns the most relevant sentence from context."""
    sentences = [s.strip() for s in context.replace("\n", " ").split(".") if len(s.strip()) > 20]
    if not sentences:
        return context[:500]

    q_tokens = set(query.lower().split())
    scored = []
    for sent in sentences:
        s_tokens = set(sent.lower().split())
        overlap = len(q_tokens & s_tokens) / max(len(q_tokens), 1)
        scored.append((overlap, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for _, s in scored[:3]]
    return ". ".join(top_sentences) + "."
