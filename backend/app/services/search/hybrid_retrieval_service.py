from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.models import KnowledgeDocument, SnakeImage, SnakeSpecies
from backend.app.services.rag.embedding_service import embed_query
from backend.app.services.rag.tools.reranker import rerank_species_profiles
from backend.app.services.search.image_retrieval_service import search_by_description

DESCRIPTION_SECTIONS = ["identification", "habitat_behavior", "distribution"]


def search_species_knowledge(db: Session, description: str, top_k: int) -> list[dict]:
    """Tìm species bằng semantic search trên knowledge documents."""
    query_embedding = embed_query(description)
    distance = KnowledgeDocument.embedding.cosine_distance(query_embedding).label("distance")

    stmt = (
        select(KnowledgeDocument, SnakeSpecies, distance)
        .join(SnakeSpecies, KnowledgeDocument.species_id == SnakeSpecies.id)
        .where(
            KnowledgeDocument.embedding.is_not(None),
            KnowledgeDocument.species_id.is_not(None),
            KnowledgeDocument.section.in_(DESCRIPTION_SECTIONS),
        )
        .order_by(distance)
        .limit(top_k * settings.SEARCH_PROFILE_CHUNK_K)
    )

    rows = db.execute(stmt).all()
    species_results = {}

    for document, species, dist in rows:
        score = 1 - float(dist)

        if species.id not in species_results:
            species_results[species.id] = {
                "species_id": species.id,
                "binomial_name": species.binomial_name,
                "vietnamese_name": species.vietnamese_name,
                "is_mivs": species.is_mivs,
                "knowledge_score": score,
            }

    results = list(species_results.values())
    results.sort(key=lambda item: item["knowledge_score"], reverse=True)
    return results[:top_k]


def hybrid_search(db: Session, description: str, focus: Literal["visual", "mixed", "contextual"], search_mode: Literal["fast", "deep"], top_k: int) -> list[dict]:
    """Kết hợp SigLIP và BGE bằng Weighted RRF."""
    candidate_k = settings.SEARCH_FAST_CANDIDATE_K if search_mode == "fast" else settings.SEARCH_DEEP_CANDIDATE_K

    image_results = search_by_description(db, description, top_k=candidate_k)
    knowledge_results = search_species_knowledge(db, description, top_k=candidate_k)

    image_weight, knowledge_weight = _get_search_weights(focus)
    candidates = _weighted_rrf(image_results, knowledge_results, image_weight, knowledge_weight)
    _attach_species_metadata(db, candidates)

    if search_mode == "fast":
        _apply_mivs_boost(candidates, "rrf_score")
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return _prepare_results(db, candidates[:top_k])

    candidates = candidates[:settings.SEARCH_DEEP_FUSION_TOP_K]
    _build_species_profiles(db, description, candidates)
    candidates = rerank_species_profiles(description, candidates)

    _apply_mivs_boost(candidates, "rerank_score")
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return _prepare_results(db, candidates[:top_k])


def _weighted_rrf(image_results: list[dict], knowledge_results: list[dict], image_weight: float, knowledge_weight: float) -> list[dict]:
    """Fusion hai danh sách species dựa trên rank thay vì raw similarity."""
    candidates = {}

    for rank, result in enumerate(image_results, start=1):
        species_id = result["species_id"]
        item = candidates.setdefault(species_id, {"species_id": species_id})
        item.update(result)
        item["image_rank"] = rank

    for rank, result in enumerate(knowledge_results, start=1):
        species_id = result["species_id"]
        item = candidates.setdefault(species_id, {"species_id": species_id})
        item.update(result)
        item["knowledge_rank"] = rank

    for item in candidates.values():
        score = 0.0
        if item.get("image_rank"):
            score += image_weight / (settings.SEARCH_RRF_K + item["image_rank"])
        if item.get("knowledge_rank"):
            score += knowledge_weight / (settings.SEARCH_RRF_K + item["knowledge_rank"])
        item["rrf_score"] = score

    results = list(candidates.values())
    results.sort(key=lambda item: item["rrf_score"], reverse=True)
    return results


def _get_search_weights(focus: str) -> tuple[float, float]:
    """Chọn trọng số SigLIP và BGE theo loại description."""
    if focus == "visual":
        return settings.SEARCH_VISUAL_IMAGE_WEIGHT, settings.SEARCH_VISUAL_KNOWLEDGE_WEIGHT
    if focus == "contextual":
        return settings.SEARCH_CONTEXTUAL_IMAGE_WEIGHT, settings.SEARCH_CONTEXTUAL_KNOWLEDGE_WEIGHT
    return settings.SEARCH_MIXED_IMAGE_WEIGHT, settings.SEARCH_MIXED_KNOWLEDGE_WEIGHT


def _attach_species_metadata(db: Session, candidates: list[dict]) -> None:
    """Bổ sung metadata chuẩn từ bảng species."""
    if not candidates:
        return

    species_ids = [candidate["species_id"] for candidate in candidates]
    species_list = db.query(SnakeSpecies).filter(SnakeSpecies.id.in_(species_ids)).all()
    species_map = {species.id: species for species in species_list}

    for candidate in candidates:
        species = species_map[candidate["species_id"]]
        candidate["binomial_name"] = species.binomial_name
        candidate["vietnamese_name"] = species.vietnamese_name
        candidate["is_mivs"] = species.is_mivs


def _build_species_profiles(db: Session, description: str, candidates: list[dict]) -> None:
    """Build profile từ các knowledge chunk phù hợp nhất của mỗi species."""
    if not candidates:
        return

    species_ids = [candidate["species_id"] for candidate in candidates]
    query_embedding = embed_query(description)
    distance = KnowledgeDocument.embedding.cosine_distance(query_embedding).label("distance")

    stmt = (
        select(KnowledgeDocument, distance)
        .where(
            KnowledgeDocument.species_id.in_(species_ids),
            KnowledgeDocument.embedding.is_not(None),
            KnowledgeDocument.section.in_(DESCRIPTION_SECTIONS),
        )
        .order_by(distance)
    )

    rows = db.execute(stmt).all()
    chunks = {species_id: [] for species_id in species_ids}

    for document, dist in rows:
        species_chunks = chunks[document.species_id]
        if len(species_chunks) >= settings.SEARCH_PROFILE_CHUNK_K:
            continue

        species_chunks.append({
            "section": document.section,
            "content": document.content,
            "score": 1 - float(dist),
        })

    for candidate in candidates:
        candidate["profile"] = _format_species_profile(candidate, chunks[candidate["species_id"]])


def _format_species_profile(candidate: dict, chunks: list[dict]) -> str:
    """Gom metadata và knowledge thành profile cho reranker."""
    parts = [
        f"Tên khoa học: {candidate['binomial_name']}",
        f"Tên tiếng Việt: {candidate['vietnamese_name'] or ''}",
    ]

    for chunk in chunks:
        parts.append(f"{chunk['section']}: {chunk['content']}")

    return "\n".join(parts)


def _apply_mivs_boost(candidates: list[dict], score_key: str) -> None:
    """Boost nhẹ MIVS sau khi retrieval/reranking hoàn tất."""
    for candidate in candidates:
        score = candidate[score_key]
        if candidate["is_mivs"]:
            score *= settings.SEARCH_MIVS_FACTOR
        candidate["score"] = score


def _prepare_results(db: Session, candidates: list[dict]) -> list[dict]:
    """Chuẩn hoá kết quả cuối để trả về API."""
    if not candidates:
        return []

    species_ids = [candidate["species_id"] for candidate in candidates]
    images = db.query(SnakeImage).filter(SnakeImage.species_id.in_(species_ids)).all()
    image_map = {}

    for image in images:
        image_map.setdefault(image.species_id, image.image_url)

    results = []
    for candidate in candidates:
        results.append({
            "species_id": candidate["species_id"],
            "binomial_name": candidate["binomial_name"],
            "vietnamese_name": candidate["vietnamese_name"],
            "image_url": candidate.get("image_url") or image_map.get(candidate["species_id"]),
            "is_mivs": candidate["is_mivs"],
            "score": candidate["score"],
        })

    return results