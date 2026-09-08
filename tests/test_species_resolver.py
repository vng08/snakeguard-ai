from types import SimpleNamespace

from backend.app.services.rag.tools import species_resolver


class FakeQuery:
    """Giả lập query lấy danh sách species."""

    def __init__(self, species):
        self.species = species

    def all(self):
        return self.species


class FakeDB:
    """Giả lập database session."""

    def __init__(self, species):
        self.species = species

    def query(self, *args):
        return FakeQuery(self.species)


def make_species():
    """Tạo dữ liệu species giả phục vụ test."""
    return [
        SimpleNamespace(id=1, binomial_name="Naja kaouthia", vietnamese_name="Rắn hổ đất, Rắn hổ mang một mắt kính", genus="Naja", family="Elapidae"),
        SimpleNamespace(id=2, binomial_name="Naja siamensis", vietnamese_name="Rắn hổ mang Xiêm", genus="Naja", family="Elapidae"),
        SimpleNamespace(id=3, binomial_name="Bungarus fasciatus", vietnamese_name="Rắn cạp nong", genus="Bungarus", family="Elapidae"),
    ]


def test_exact_match():
    db = FakeDB(make_species())

    result = species_resolver.resolve_species(db, "Rắn cạp nong")

    assert result["status"] == "resolved"
    assert result["resolved_species_ids"] == [3]
    assert result["candidates"][0]["match_type"] == "exact"
    assert result["candidates"][0]["lexical_score"] == 100.0
    assert result["candidates"][0]["semantic_score"] is None


def test_partial_match_is_ambiguous():
    db = FakeDB(make_species())

    result = species_resolver.resolve_species(db, "Rắn hổ mang")

    assert result["status"] == "ambiguous"
    assert len(result["candidates"]) == 2
    assert result["resolved_species_ids"] == []


def test_fuzzy_match_fast():
    db = FakeDB(make_species())

    result = species_resolver.resolve_species(db, "Rắn cạp nog", search_mode="fast")

    assert result["status"] == "resolved"
    assert result["resolved_species_ids"] == [3]
    assert result["candidates"][0]["match_type"] == "fuzzy"
    assert result["candidates"][0]["lexical_score"] is not None
    assert result["candidates"][0]["semantic_score"] is None


def test_fast_mode_does_not_use_deep_tools(monkeypatch):
    db = FakeDB(make_species())

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Deep matching không được gọi trong Fast mode.")

    monkeypatch.setattr(species_resolver, "_find_wratio_matches", fail_if_called)
    monkeypatch.setattr(species_resolver, "_find_semantic_matches", fail_if_called)
    monkeypatch.setattr(species_resolver, "rerank_species", fail_if_called)

    result = species_resolver.resolve_species(db, "yellow banded snake", search_mode="fast")

    assert result["status"] == "not_found"


def test_deep_confident_ratio_skips_deep_tools(monkeypatch):
    db = FakeDB(make_species())

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Deep pipeline không được gọi khi Ratio đã đủ chắc chắn.")

    monkeypatch.setattr(species_resolver, "_find_wratio_matches", fail_if_called)
    monkeypatch.setattr(species_resolver, "_find_semantic_matches", fail_if_called)
    monkeypatch.setattr(species_resolver, "rerank_species", fail_if_called)

    result = species_resolver.resolve_species(db, "Rắn hổ dap", search_mode="deep")

    assert result["status"] == "resolved"
    assert result["resolved_species_ids"] == [1]
    assert result["candidates"][0]["match_type"] == "fuzzy"
    assert result["candidates"][0]["score"] >= 88


def test_deep_mode_uses_wratio_semantic_and_reranker(monkeypatch):
    db = FakeDB(make_species())

    def mock_wratio(query, entries):
        entry = next(item for item in entries if item["species"].id == 1)
        return [species_resolver._candidate(entry, 88.0, "wratio")]

    def mock_semantic(query, entries):
        entry = next(item for item in entries if item["species"].id == 3)
        return [species_resolver._candidate(entry, 0.91, "semantic")]

    def mock_reranker(query, candidates):
        candidate = next(item for item in candidates if item["species_id"] == 3)
        candidate["match_score"] = 0.955
        candidate["rerank_score"] = 0.95
        candidate["final_score"] = 0.953
        return [candidate]

    monkeypatch.setattr(species_resolver, "_find_wratio_matches", mock_wratio)
    monkeypatch.setattr(species_resolver, "_find_semantic_matches", mock_semantic)
    monkeypatch.setattr(species_resolver, "rerank_species", mock_reranker)

    result = species_resolver.resolve_species(db, "yellow banded snake", search_mode="deep")

    assert result["status"] == "resolved"
    assert result["resolved_species_ids"] == [3]
    assert result["candidates"][0]["semantic_score"] == 0.91
    assert result["candidates"][0]["final_score"] == 0.953


def test_deep_reranker_ambiguous(monkeypatch):
    db = FakeDB(make_species())

    def mock_wratio(query, entries):
        first = next(item for item in entries if item["species"].id == 1)
        second = next(item for item in entries if item["species"].id == 2)
        return [
            species_resolver._candidate(first, 90.0, "wratio"),
            species_resolver._candidate(second, 89.0, "wratio"),
        ]

    def mock_semantic(query, entries):
        return []

    def mock_reranker(query, candidates):
        candidates[0]["match_score"] = 0.90
        candidates[0]["rerank_score"] = 0.90
        candidates[0]["final_score"] = 0.90

        candidates[1]["match_score"] = 0.89
        candidates[1]["rerank_score"] = 0.87
        candidates[1]["final_score"] = 0.882

        return candidates[:2]

    monkeypatch.setattr(species_resolver, "_find_wratio_matches", mock_wratio)
    monkeypatch.setattr(species_resolver, "_find_semantic_matches", mock_semantic)
    monkeypatch.setattr(species_resolver, "rerank_species", mock_reranker)

    result = species_resolver.resolve_species(db, "cobra snake", search_mode="deep")

    assert result["status"] == "ambiguous"
    assert result["resolved_species_ids"] == []


def test_merge_candidates_keeps_lexical_and_semantic_scores():
    entries = species_resolver._build_name_entries(make_species())
    entry = next(item for item in entries if item["species"].id == 1)

    lexical = species_resolver._candidate(entry, 90.0, "wratio")
    semantic = species_resolver._candidate(entry, 0.74, "semantic")

    candidates = species_resolver._merge_candidates([lexical], [semantic])

    assert len(candidates) == 1
    assert candidates[0]["species_id"] == 1
    assert candidates[0]["lexical_score"] == 90.0
    assert candidates[0]["semantic_score"] == 0.74


def test_merge_candidates_keeps_best_lexical_score():
    entries = species_resolver._build_name_entries(make_species())
    entry = next(item for item in entries if item["species"].id == 1)

    fuzzy = species_resolver._candidate(entry, 82.0, "fuzzy")
    wratio = species_resolver._candidate(entry, 90.0, "wratio")

    candidates = species_resolver._merge_candidates([fuzzy], [wratio])

    assert len(candidates) == 1
    assert candidates[0]["lexical_score"] == 90.0
    assert candidates[0]["score"] == 90.0
    assert candidates[0]["match_type"] == "wratio"