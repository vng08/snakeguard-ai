import math

from sentence_transformers import CrossEncoder

from backend.app.core.config import settings

_reranker_model = None


def get_reranker_model() -> CrossEncoder:
    """Lazy-load reranker model và dùng chung trong toàn bộ Agentic RAG."""
    global _reranker_model

    if _reranker_model is None:
        _reranker_model = CrossEncoder(settings.RERANKER_MODEL_NAME)

    return _reranker_model


def rerank_species(query: str, candidates: list[dict]) -> list[dict]:
    """Rerank species candidate và kết hợp các tín hiệu matching."""
    if not candidates:
        return []

    model = get_reranker_model()
    pairs = []

    for candidate in candidates:
        text = (
            f"Tên khoa học: {candidate['binomial_name']}. "
            f"Tên tiếng Việt: {candidate['vietnamese_name'] or ''}. "
            f"Chi: {candidate['genus'] or ''}. "
            f"Họ: {candidate['family'] or ''}."
        )
        pairs.append((query, text))

    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["match_score"] = _get_match_score(candidate)
        candidate["rerank_score"] = round(_sigmoid(float(score)), 4)
        candidate["final_score"] = round(
            candidate["match_score"] * settings.SPECIES_MATCH_WEIGHT
            + candidate["rerank_score"] * settings.SPECIES_RERANK_WEIGHT,
            4,
        )

    candidates.sort(key=lambda item: item["final_score"], reverse=True)
    return candidates[:settings.SPECIES_RERANK_TOP_K]


def rerank_documents(query: str, documents: list[dict]) -> list[dict]:
    """Rerank knowledge chunks trong Deep retrieval."""
    if not documents:
        return []

    model = get_reranker_model()
    pairs = [(query, document["content"]) for document in documents]
    scores = model.predict(pairs)

    for document, score in zip(documents, scores):
        document["rerank_score"] = round(_sigmoid(float(score)), 4)

    documents.sort(key=lambda item: item["rerank_score"], reverse=True)
    return documents[:settings.RAG_DEEP_TOP_K]


def rerank_species_profiles(query: str, candidates: list[dict]) -> list[dict]:
    """Rerank species dựa trên profile knowledge đầy đủ của từng loài."""
    if not candidates:
        return []

    model = get_reranker_model()
    pairs = [(query, candidate["profile"]) for candidate in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = round(_sigmoid(float(score)), 4)

    candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
    return candidates


def _get_match_score(candidate: dict) -> float:
    """Kết hợp lexical và semantic score sau khi chuẩn hoá về 0-1."""
    scores = []

    if candidate.get("lexical_score") is not None:
        scores.append(_normalize_lexical_score(candidate["lexical_score"]))

    if candidate.get("semantic_score") is not None:
        scores.append(_normalize_semantic_score(candidate["semantic_score"]))

    return round(max(scores), 4) if scores else 0.0

def _normalize_lexical_score(score: float) -> float:
    """Chuẩn hoá RapidFuzz score từ 0-100 về 0-1."""
    return min(max(float(score) / 100, 0), 1)


def _normalize_semantic_score(score: float) -> float:
    """Chuẩn hoá cosine similarity từ -1..1 về 0-1."""
    return min(max((float(score) + 1) / 2, 0), 1)


def _sigmoid(value: float) -> float:
    """Đưa reranker logit về khoảng 0-1."""
    value = max(min(value, 50), -50)
    return 1 / (1 + math.exp(-value))