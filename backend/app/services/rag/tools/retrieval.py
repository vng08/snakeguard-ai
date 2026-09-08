from typing import Literal

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.models import KnowledgeDocument
from backend.app.services.rag.embedding_service import embed_query


def retrieve_knowledge(db: Session, query: str, route: Literal["global", "species"], species: dict | None = None, search_mode: Literal["fast", "deep"] = "fast") -> list[dict]:
    """Tìm các knowledge chunks phù hợp bằng vector retrieval."""
    query_embedding = embed_query(query)
    candidate_k = settings.RAG_FAST_TOP_K if search_mode == "fast" else settings.RAG_DEEP_CANDIDATE_K

    if route == "global" or not species:
        return _search_global(db, query_embedding, candidate_k)

    return _search_species_scopes(db, query_embedding, species, candidate_k)


def _search_species_scopes(db: Session, query_embedding: list[float], species: dict, candidate_k: int) -> list[dict]:
    """Vector search theo species → group → genus → family → general."""
    species_id = species["species_id"]
    binomial_name = species["binomial_name"]
    genus = species.get("genus")
    family = species.get("family")

    species_condition = or_( KnowledgeDocument.species_id == species_id, (KnowledgeDocument.scope_type == "species") & (KnowledgeDocument.scope_value == binomial_name))
    scopes = [("species", species_condition), ("group", (KnowledgeDocument.scope_type == "group") & KnowledgeDocument.scope_value.contains(binomial_name))]

    if genus:
        scopes.append(("genus", (KnowledgeDocument.scope_type == "genus") & (KnowledgeDocument.scope_value == genus)))

    if family:
        scopes.append(("family", (KnowledgeDocument.scope_type == "family") & (KnowledgeDocument.scope_value == family)))

    scopes.append(("general", KnowledgeDocument.scope_type == "general"))

    results = []

    # Search từng scope để không scope nào bị loại trước khi so semantic score
    for match_level, condition in scopes:
        results.extend(_search_level(db, query_embedding, condition, candidate_k, match_level))

    results = _deduplicate_results(results)
    results.sort(key=lambda item: item["vector_score"], reverse=True)

    return results[:candidate_k]


def _search_level(db: Session, query_embedding: list[float], condition, limit: int, match_level: str) -> list[dict]:
    """Vector search trong một taxonomy scope."""
    distance = KnowledgeDocument.embedding.cosine_distance(query_embedding).label("distance")
    stmt = (select(KnowledgeDocument, distance) .where(KnowledgeDocument.embedding.is_not(None), condition) .order_by(distance) .limit(limit))
    rows = db.execute(stmt).all()

    return [_build_result(document, distance_value, match_level) for document, distance_value in rows]


def _search_global(db: Session, query_embedding: list[float], limit: int) -> list[dict]:
    """Vector search trong knowledge chung, không gắn với loài hoặc taxonomy cụ thể."""
    distance = KnowledgeDocument.embedding.cosine_distance(query_embedding).label("distance")

    stmt = (select(KnowledgeDocument, distance).where(KnowledgeDocument.embedding.is_not(None), KnowledgeDocument.scope_type == "general").order_by(distance).limit(limit))

    rows = db.execute(stmt).all()
    return [_build_result(document, distance_value, "global") for document, distance_value in rows]


def _build_result(document: KnowledgeDocument, distance: float, match_level: str) -> dict:
    """Chuẩn hoá một knowledge chunk thành retrieval result."""
    return {
        "id": document.id,
        "species_id": document.species_id,
        "document_type": document.document_type,
        "scope_type": document.scope_type,
        "scope_value": document.scope_value,
        "section": document.section,
        "subsection": document.subsection,
        "content": document.content,
        "source": document.source,
        "source_url": document.source_url,
        "vector_score": 1 - float(distance),
        "match_level": match_level,
    }


def _deduplicate_results(results: list[dict]) -> list[dict]:
    """Loại chunk trùng khi một document match nhiều scope."""
    best_results = {}

    for result in results:
        current = best_results.get(result["id"])

        if current is None or result["vector_score"] > current["vector_score"]:
            best_results[result["id"]] = result

    return list(best_results.values())