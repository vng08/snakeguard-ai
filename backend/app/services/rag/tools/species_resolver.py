import re
import unicodedata
from typing import Literal

import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.models import SnakeSpecies
from backend.app.services.rag.embedding_service import get_embedding_model
from backend.app.services.rag.tools.reranker import rerank_species

_species_embedding_cache = None


def resolve_species(db: Session, species_text: str, search_mode: Literal["fast", "deep"] = "fast") -> dict:
    """Resolve tên rắn thành species trong database."""
    query = _normalize_name(species_text)

    if not query:
        return _build_result("not_found", [])

    species_list = db.query(SnakeSpecies).all()
    name_entries = _build_name_entries(species_list)

    # Ưu tiên tên khớp chính xác
    exact_candidates = _find_exact_matches(query, name_entries)
    if exact_candidates:
        return _finalize_candidates(exact_candidates)

    # Kiểm tra query là một phần của tên hoặc alias
    partial_candidates = _find_partial_matches(query, name_entries)
    if partial_candidates and (search_mode == "fast" or len(partial_candidates) == 1):
        return _finalize_candidates(partial_candidates)

    # Ratio xử lý typo và resolve ngay nếu có một candidate đủ chắc chắn
    fuzzy_candidates = _find_fuzzy_matches(query, name_entries)
    if fuzzy_candidates and (search_mode == "fast" or _is_confident_match(fuzzy_candidates)):
        return _finalize_candidates(fuzzy_candidates)

    if search_mode == "fast":
        return _build_result("not_found", [])

    # Deep mode mở rộng candidate bằng WRatio và BGE rồi dùng reranker
    wratio_candidates = _find_wratio_matches(query, name_entries)
    semantic_candidates = _find_semantic_matches(species_text, name_entries)
    candidates = _merge_candidates(partial_candidates, fuzzy_candidates, wratio_candidates, semantic_candidates)

    if not candidates:
        return _build_result("not_found", [])

    reranked_candidates = rerank_species(species_text, candidates)
    return _finalize_reranked_candidates(reranked_candidates)


def _build_name_entries(species_list: list[SnakeSpecies]) -> list[dict]:
    """Tạo danh sách tên khoa học, tên Việt và alias dùng cho matching."""
    entries = []

    for species in species_list:
        names = [species.binomial_name]

        if species.vietnamese_name:
            names.extend(_split_aliases(species.vietnamese_name))

        for name in names:
            entries.append({"species": species, "name": name, "normalized_name": _normalize_name(name)})

    return entries


def _split_aliases(value: str) -> list[str]:
    """Tách các tên hoặc alias được lưu chung trong một field."""
    return [name.strip() for name in re.split(r"[,;/|]", value) if name.strip()]


def _find_exact_matches(query: str, entries: list[dict]) -> list[dict]:
    """Tìm tên khớp hoàn toàn với query."""
    matches = []

    for entry in entries:
        if entry["normalized_name"] == query:
            matches.append(_candidate(entry, 100.0, "exact"))

    return _deduplicate_candidates(matches)


def _find_partial_matches(query: str, entries: list[dict]) -> list[dict]:
    """Tìm query nằm trong tên loài hoặc alias."""
    matches = []

    for entry in entries:
        name = entry["normalized_name"]

        if query in name or name in query:
            score = fuzz.ratio(query, name)
            matches.append(_candidate(entry, score, "partial"))

    return _deduplicate_candidates(matches)


def _find_fuzzy_matches(query: str, entries: list[dict]) -> list[dict]:
    """Dùng Ratio để tìm tên gần giống và xử lý lỗi chính tả."""
    matches = []

    for entry in entries:
        score = fuzz.ratio(query, entry["normalized_name"])

        if score >= settings.SPECIES_FUZZY_THRESHOLD:
            matches.append(_candidate(entry, score, "fuzzy"))

    matches = _deduplicate_candidates(matches)

    if not matches:
        return []

    best_score = matches[0]["score"]
    return [candidate for candidate in matches if candidate["score"] >= best_score - settings.SPECIES_AMBIGUITY_MARGIN]


def _find_wratio_matches(query: str, entries: list[dict]) -> list[dict]:
    """Dùng WRatio để mở rộng candidate trong Deep mode."""
    matches = []

    for entry in entries:
        score = fuzz.WRatio(query, entry["normalized_name"])

        if score >= settings.SPECIES_FUZZY_THRESHOLD:
            matches.append(_candidate(entry, score, "wratio"))

    matches = _deduplicate_candidates(matches)
    return matches[:settings.SPECIES_DEEP_CANDIDATE_K]


def _find_semantic_matches(query: str, entries: list[dict]) -> list[dict]:
    """Dùng BGE để tạo semantic candidates trong Deep mode."""
    model = get_embedding_model()
    embeddings = _get_species_embeddings(model, entries)

    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    scores = np.dot(embeddings, query_embedding)

    ranked_indices = np.argsort(scores)[::-1]
    matches = []
    seen_species = set()

    for index in ranked_indices:
        score = float(scores[index])
        species_id = entries[index]["species"].id

        if score < settings.SPECIES_SEMANTIC_THRESHOLD:
            break

        if species_id in seen_species:
            continue

        matches.append(_candidate(entries[index], score, "semantic"))
        seen_species.add(species_id)

        if len(matches) >= settings.SPECIES_DEEP_CANDIDATE_K:
            break

    return matches


def _get_species_embeddings(model: SentenceTransformer, entries: list[dict]) -> np.ndarray:
    """Embed tên loài một lần và cache trong RAM."""
    global _species_embedding_cache

    names = [entry["name"] for entry in entries]
    cache_key = tuple((entry["species"].id, entry["name"]) for entry in entries)

    if _species_embedding_cache is None or _species_embedding_cache["key"] != cache_key:
        embeddings = model.encode(names, normalize_embeddings=True)
        _species_embedding_cache = {"key": cache_key, "embeddings": embeddings}

    return _species_embedding_cache["embeddings"]


def _candidate(entry: dict, score: float, match_type: str) -> dict:
    """Chuyển kết quả match thành candidate thống nhất."""
    species = entry["species"]
    is_semantic = match_type == "semantic"

    return {
        "species_id": species.id,
        "binomial_name": species.binomial_name,
        "vietnamese_name": species.vietnamese_name,
        "genus": species.genus,
        "family": species.family,
        "matched_name": entry["name"],
        "score": round(float(score), 4),
        "match_type": match_type,
        "lexical_score": None if is_semantic else round(float(score), 4),
        "semantic_score": round(float(score), 4) if is_semantic else None,
    }


def _merge_candidates(*candidate_groups: list[dict]) -> list[dict]:
    """Gộp candidate và giữ lại cả lexical lẫn semantic evidence."""
    merged = {}

    for candidates in candidate_groups:
        for candidate in candidates:
            species_id = candidate["species_id"]

            if species_id not in merged:
                merged[species_id] = candidate.copy()
                continue

            current = merged[species_id]

            lexical_score = candidate["lexical_score"]
            if lexical_score is not None and (current["lexical_score"] is None or lexical_score > current["lexical_score"]):
                current["lexical_score"] = lexical_score
                current["matched_name"] = candidate["matched_name"]
                current["score"] = candidate["score"]
                current["match_type"] = candidate["match_type"]

            semantic_score = candidate["semantic_score"]
            if semantic_score is not None and (current["semantic_score"] is None or semantic_score > current["semantic_score"]):
                current["semantic_score"] = semantic_score

    return list(merged.values())


def _deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    """Giữ candidate có score cao nhất cho mỗi species."""
    best_by_species = {}

    for candidate in candidates:
        species_id = candidate["species_id"]
        current = best_by_species.get(species_id)

        if current is None or candidate["score"] > current["score"]:
            best_by_species[species_id] = candidate

    return sorted(best_by_species.values(), key=lambda item: item["score"], reverse=True)


def _is_confident_match(candidates: list[dict]) -> bool:
    """Kiểm tra Ratio có đủ chắc chắn để resolve ngay."""
    return len(candidates) == 1 and candidates[0]["score"] >= settings.SPECIES_DEEP_CONFIDENCE_THRESHOLD


def _finalize_reranked_candidates(candidates: list[dict]) -> dict:
    """Quyết định kết quả dựa trên final score."""
    if not candidates or candidates[0]["final_score"] < settings.SPECIES_FINAL_THRESHOLD:
        return _build_result("not_found", candidates)

    if len(candidates) == 1:
        return _build_result("resolved", candidates)

    score_gap = candidates[0]["final_score"] - candidates[1]["final_score"]

    if score_gap >= settings.SPECIES_FINAL_AMBIGUITY_MARGIN:
        return _build_result("resolved", [candidates[0]])

    return _build_result("ambiguous", candidates)


def _finalize_candidates(candidates: list[dict]) -> dict:
    """Xác định kết quả đã resolve hay vẫn còn mơ hồ."""
    if len(candidates) == 1:
        return _build_result("resolved", candidates)

    return _build_result("ambiguous", candidates)


def _build_result(status: Literal["resolved", "ambiguous", "not_found"], candidates: list[dict]) -> dict:
    """Tạo output chuẩn cho Species Resolver."""
    return {"status": status, "candidates": candidates, "resolved_species_ids": [candidates[0]["species_id"]] if status == "resolved" else []}


def _normalize_name(value: str) -> str:
    """Chuẩn hoá tên để so sánh không phân biệt hoa thường và dấu."""
    value = value.lower().strip()
    value = unicodedata.normalize("NFD", value)
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")

    value = value.replace("đ", "d")
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()