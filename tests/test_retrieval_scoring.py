from backend.app.services.image_retrieval_service import calculate_species_score
from backend.app.services.hybrid_retrieval_service import normalize_scores
import pytest

# Test weighted Top-3 của image retrieval
def test_calculate_species_score():
    matches = [
        (0.15, "image_3", "species"),
        (0.22, "image_1", "species"),
        (0.18, "image_2", "species"),
    ]

    score, best_match = calculate_species_score(matches)
    expected_score = 0.6 * 0.22 + 0.3 * 0.18 + 0.1 * 0.15

    assert score == expected_score
    assert best_match[0] == 0.22
    assert best_match[1] == "image_1"


# Test normalize score về khoảng 0-1
def test_normalize_scores():
    results = [
        {"species_id": 1, "similarity": 0.10},
        {"species_id": 2, "similarity": 0.20},
        {"species_id": 3, "similarity": 0.30},
    ]

    normalized = normalize_scores(results, "similarity")

    assert normalized[1] == pytest.approx(0.0)
    assert normalized[2] == pytest.approx(0.5)
    assert normalized[3] == pytest.approx(1.0)