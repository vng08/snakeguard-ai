from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import KnowledgeDocument, SnakeImage, SnakeSpecies
from backend.app.services.image_retrieval_service import search_by_description
from backend.app.services.retrieval_service import embed_query


CANDIDATE_K = 20
IMAGE_WEIGHT = 0.5
KNOWLEDGE_WEIGHT = 0.5
MIVS_FACTOR = 1.05
DESCRIPTION_SECTIONS = ["identification", "habitat_behavior", "distribution"]


def normalize_scores(results, score_key):
    if not results:
        return {}

    scores = [result[score_key] for result in results]
    min_score, max_score = min(scores), max(scores)

    if max_score == min_score:
        return {result["species_id"]: 1.0 for result in results}

    return {
        result["species_id"]: (result[score_key] - min_score) / (max_score - min_score)
        for result in results
    }


def search_species_knowledge(db: Session, description: str, top_k: int = CANDIDATE_K):
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
        .limit(top_k * 5)
    )

    rows = db.execute(stmt).all()
    results, seen_species = [], set()

    for document, species, dist in rows:
        if species.id in seen_species:
            continue

        seen_species.add(species.id)
        results.append({
            "species_id": species.id,
            "binomial_name": species.binomial_name,
            "vietnamese_name": species.vietnamese_name,
            "is_mivs": species.is_mivs,
            "knowledge_score": 1 - float(dist),
        })

        if len(results) >= top_k:
            break

    return results


def hybrid_search(db: Session, description: str, top_k: int = 5):
    image_results = search_by_description(db, description, top_k=CANDIDATE_K)
    knowledge_results = search_species_knowledge(db, description, top_k=CANDIDATE_K)

    image_scores = normalize_scores(image_results, "similarity")
    knowledge_scores = normalize_scores(knowledge_results, "knowledge_score")

    metadata = {}

    for result in image_results:
        metadata.setdefault(result["species_id"], {}).update(result)

    for result in knowledge_results:
        metadata.setdefault(result["species_id"], {}).update(result)

    species_ids = list(metadata.keys())
    species_list = db.query(SnakeSpecies).filter(SnakeSpecies.id.in_(species_ids)).all()
    species_map = {species.id: species for species in species_list}

    results = []

    for species_id, item in metadata.items():
        species = species_map[species_id]
        image_score = image_scores.get(species_id, 0.0)
        knowledge_score = knowledge_scores.get(species_id, 0.0)

        if species.is_mivs:
            image_score *= MIVS_FACTOR
            knowledge_score *= MIVS_FACTOR

        hybrid_score = IMAGE_WEIGHT * image_score + KNOWLEDGE_WEIGHT * knowledge_score

        if "image_url" not in item:
            image = db.query(SnakeImage).filter(SnakeImage.species_id == species_id).first()
            item["image_url"] = image.image_url if image else None

        item["image_score_normalized"] = image_score
        item["knowledge_score_normalized"] = knowledge_score
        item["is_mivs"] = species.is_mivs
        item["hybrid_score"] = hybrid_score
        results.append(item)

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results[:top_k]