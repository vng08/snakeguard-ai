from collections import defaultdict
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.models import SnakeImage, SnakeSpecies
from ml.image_retrieval.siglip_encoder import SigLIPEncoder


_encoder = None


def get_encoder() -> SigLIPEncoder:
    """Lazy-load SigLIP encoder dùng cho image retrieval."""
    global _encoder
    if _encoder is None:
        _encoder = SigLIPEncoder()
    return _encoder


def calculate_species_score(matches: list) -> tuple:
    """Tính score species từ các ảnh match tốt nhất."""
    top_k = len(settings.SEARCH_TOP_IMAGE_WEIGHTS)
    best_matches = sorted(matches, key=lambda item: item[0], reverse=True)[:top_k]
    weights = settings.SEARCH_TOP_IMAGE_WEIGHTS[:len(best_matches)]
    weight_sum = sum(weights)

    score = sum(match[0] * weight for match, weight in zip(best_matches, weights)) / weight_sum
    return score, best_matches[0]


def search_by_description(db: Session, description: str, top_k: int = 5) -> list[dict]:
    """Tìm species bằng similarity giữa description và image embedding."""
    query_embedding = get_encoder().encode_text(description)
    distance = SnakeImage.embedding.cosine_distance(query_embedding)

    rows = (
        db.query(SnakeImage, SnakeSpecies, distance.label("distance"))
        .join(SnakeSpecies, SnakeImage.species_id == SnakeSpecies.id)
        .filter(SnakeImage.embedding.isnot(None))
        .order_by(distance)
        .all()
    )

    grouped = defaultdict(list)

    for image, species, dist in rows:
        grouped[species.id].append((1 - float(dist), image, species))

    results = []

    for matches in grouped.values():
        score, best_match = calculate_species_score(matches)
        _, best_image, species = best_match

        results.append({
            "species_id": species.id,
            "binomial_name": species.binomial_name,
            "vietnamese_name": species.vietnamese_name,
            "similarity": score,
            "image_url": best_image.image_url,
        })

    results.sort(key=lambda item: item["similarity"], reverse=True)
    return results[:top_k]