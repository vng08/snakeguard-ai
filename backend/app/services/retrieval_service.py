import re
import unicodedata
from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy import or_, select
from sqlalchemy.orm import Session
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.app.core.config import settings
from backend.app.db.models import KnowledgeDocument, SnakeSpecies


DEFAULT_TOP_K = 5
RERANK_CANDIDATE_K = 20
MIN_NAME_PREFIX_TOKENS = 3

SECTION_LABELS = {
    "diagnosis": "Chẩn đoán, triệu chứng, dấu hiệu",
    "treatment": "Điều trị",
    "first_aid": "Sơ cứu",
    "identification": "Đặc điểm nhận dạng",
    "distribution": "Phân bố",
    "habitat_behavior": "Môi trường sống, tập tính",
    "venom_danger": "Nọc độc, mức độ nguy hiểm",
}

SECTION_KEYWORDS = {
    "first_aid": ["sơ cứu", "so cuu", "xử lý ban đầu", "xu ly ban dau", "nên làm gì", "nen lam gi"],
    "identification": ["nhận dạng", "nhan dang", "nhận biết", "nhan biet", "đặc điểm", "dac diem", "hình dạng", "hinh dang"],
    "distribution": ["phân bố", "phan bo", "ở đâu", "o dau", "khu vực nào", "khu vuc nao"],
    "habitat_behavior": ["môi trường sống", "moi truong song", "tập tính", "tap tinh", "sinh cảnh", "sinh canh", "hoạt động", "hoat dong"],
    "venom_danger": ["nọc độc", "noc doc", "có độc", "co doc", "độc không", "doc khong", "nguy hiểm", "nguy hiem"],
    "diagnosis": ["triệu chứng", "trieu chung", "dấu hiệu", "dau hieu", "biểu hiện", "bieu hien"],
    "treatment": ["điều trị", "dieu tri", "chữa trị", "chua tri", "huyết thanh", "huyet thanh"],
}


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load embedding model một lần."""
    print(f"Loading retrieval model: {settings.EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(settings.EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_reranker_model():
    """Load reranker một lần."""
    print(f"Loading reranker model: {settings.RERANKER_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(settings.RERANKER_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(settings.RERANKER_MODEL_NAME)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    return tokenizer, model, device


def normalize_text(text):
    """Chuẩn hóa text để match tên rắn."""
    text = text.lower().replace("đ", "d")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip()


def count_name_prefix_in_query(name, query):
    """Đếm số token đầu của tên loài xuất hiện liên tiếp trong query."""
    name_tokens = normalize_text(name).split()
    query_tokens = normalize_text(query).split()
    best = 0

    for start in range(len(query_tokens)):
        count = 0

        for i, token in enumerate(name_tokens):
            if start + i >= len(query_tokens) or query_tokens[start + i] != token:
                break

            count += 1

        best = max(best, count)

    return best


def resolve_species(db: Session, query: str):
    """Resolve một hoặc nhiều species từ tên đầy đủ hoặc tên Việt rút gọn."""
    query_norm = normalize_text(query)
    species_list = db.execute(select(SnakeSpecies)).scalars().all()
    exact_matches = {}

    for species in species_list:
        for name in [species.binomial_name, species.vietnamese_name]:
            if not name:
                continue

            name_norm = normalize_text(name)

            if name_norm in query_norm:
                score = len(name_norm)

                if species.id not in exact_matches or score > exact_matches[species.id][0]:
                    exact_matches[species.id] = (score, species)

    if exact_matches:
        best_score = max(score for score, _ in exact_matches.values())
        return [species for score, species in exact_matches.values() if score == best_score]

    partial_matches = []

    for species in species_list:
        if not species.vietnamese_name:
            continue

        score = count_name_prefix_in_query(species.vietnamese_name, query)

        if score >= MIN_NAME_PREFIX_TOKENS:
            partial_matches.append((score, species))

    if not partial_matches:
        return []

    best_score = max(score for score, _ in partial_matches)
    return [species for score, species in partial_matches if score == best_score]


def detect_sections(query):
    """Xác định một hoặc nhiều loại thông tin user đang hỏi."""
    query_norm = normalize_text(query)
    return [section for section, keywords in SECTION_KEYWORDS.items() if any(normalize_text(keyword) in query_norm for keyword in keywords)]


def get_section_filters(sections):
    """Map intents sang sections trong DB."""
    filters = []

    for section in sections:
        if section == "first_aid":
            filters.extend(["first_aid", "treatment"])
        else:
            filters.append(section)

    return list(dict.fromkeys(filters))


def build_query_text(query, species=None, sections=None):
    """Thêm semantic context vào query."""
    parts = [query]

    if species:
        parts.append(f"Species: {species.binomial_name}")

        if species.vietnamese_name:
            parts.append(f"Tên tiếng Việt: {species.vietnamese_name}")

    if sections:
        labels = [SECTION_LABELS[section] for section in sections]
        parts.append(f"Sections: {'; '.join(labels)}")

    return "\n".join(parts)


def build_rerank_text(result):
    """Tạo passage có context cho reranker."""
    scope = result["scope_value"] or "Rắn cắn nói chung"
    section = SECTION_LABELS.get(result["section"], result["section"])
    parts = [f"Scope: {scope}", f"Section: {section}"]

    if result["subsection"]:
        parts.append(f"Subsection: {result['subsection']}")

    parts.extend(["", result["content"]])
    return "\n".join(parts)


def embed_query(query):
    """Embedding query thành vector."""
    model = get_embedding_model()
    embedding = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    return embedding.tolist()


def rerank_results(query, results):
    """Rerank candidates bằng BGE reranker."""
    if not results:
        return []

    tokenizer, model, device = get_reranker_model()
    queries = [query] * len(results)
    passages = [build_rerank_text(result) for result in results]
    inputs = tokenizer(queries, passages, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits.view(-1)
        scores = torch.sigmoid(logits).float().cpu().tolist()

    for result, rerank_score in zip(results, scores):
        result["vector_score"] = result["score"]
        result["rerank_score"] = float(rerank_score)
        result["score"] = float(rerank_score)

    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results


def search_level(db, query_embedding, condition, sections, limit, match_level):
    """Vector search trong một scope cụ thể."""
    distance = KnowledgeDocument.embedding.cosine_distance(query_embedding).label("distance")
    stmt = select(KnowledgeDocument, distance).where(KnowledgeDocument.embedding.is_not(None), condition)

    if sections:
        stmt = stmt.where(KnowledgeDocument.section.in_(sections))

    rows = db.execute(stmt.order_by(distance).limit(limit)).all()

    return [
        {
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
            "score": 1 - float(distance_value),
            "match_level": match_level,
        }
        for document, distance_value in rows
    ]


def search_global(db, query_embedding, sections, limit):
    """Vector search toàn KB."""
    distance = KnowledgeDocument.embedding.cosine_distance(query_embedding).label("distance")
    stmt = select(KnowledgeDocument, distance).where(KnowledgeDocument.embedding.is_not(None))

    if sections:
        stmt = stmt.where(KnowledgeDocument.section.in_(sections))

    rows = db.execute(stmt.order_by(distance).limit(limit)).all()

    return [
        {
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
            "score": 1 - float(distance_value),
            "match_level": "global",
        }
        for document, distance_value in rows
    ]


def select_top_results(results, detected_sections, top_k):
    """Chọn Top-K và giữ coverage cho nhiều intent."""
    if not detected_sections or len(detected_sections) == 1:
        return results[:top_k]

    selected, seen_ids = [], set()

    for section in detected_sections:
        result = next((item for item in results if item["section"] == section and item["id"] not in seen_ids), None)

        if result:
            selected.append(result)
            seen_ids.add(result["id"])

    for result in results:
        if len(selected) >= top_k:
            break

        if result["id"] not in seen_ids:
            selected.append(result)
            seen_ids.add(result["id"])

    return selected[:top_k]


def build_retrieval_response(results, candidates, detected_sections):
    """Chuẩn hóa kết quả và metadata của retrieval."""
    is_ambiguous = len(candidates) > 1
    resolved_species = candidates[0] if len(candidates) == 1 else None

    return {
        "results": results,
        "is_ambiguous": is_ambiguous,
        "candidate_species": [
            f"{item.vietnamese_name} ({item.binomial_name})" if item.vietnamese_name else item.binomial_name
            for item in candidates
        ] if is_ambiguous else [],
        "resolved_species": {
            "binomial_name": resolved_species.binomial_name,
            "vietnamese_name": resolved_species.vietnamese_name,
        } if resolved_species else None,
        "detected_sections": detected_sections,
    }


def search_knowledge(db: Session, query: str, top_k: int = DEFAULT_TOP_K, use_reranker: bool = True):
    """Metadata retrieval → vector search → reranking."""
    candidates = resolve_species(db, query)
    detected_sections = detect_sections(query)
    sections = get_section_filters(detected_sections)
    resolved_species = candidates[0] if len(candidates) == 1 else None

    # Không resolve được species thì search toàn KB
    if not candidates:
        query_text = build_query_text(query, sections=detected_sections)
        query_embedding = embed_query(query_text)
        results = search_global(db, query_embedding, sections, RERANK_CANDIDATE_K)

        if use_reranker:
            results = rerank_results(query_text, results)

        results = select_top_results(results, detected_sections, top_k)
        return build_retrieval_response(results, candidates, detected_sections)

    query_text = build_query_text(query, resolved_species, detected_sections)
    query_embedding = embed_query(query_text)

    species_ids = [item.id for item in candidates]
    species_names = [item.binomial_name for item in candidates]
    genera = list({item.genus for item in candidates if item.genus})
    families = list({item.family for item in candidates if item.family})
    group_conditions = [KnowledgeDocument.scope_value.contains(item.binomial_name) for item in candidates]

    # Match cả species documents và medical documents scoped theo species
    species_condition = or_(
        KnowledgeDocument.species_id.in_(species_ids),
        (KnowledgeDocument.scope_type == "species") & KnowledgeDocument.scope_value.in_(species_names),
    )

    levels = [
        ("species", species_condition),
        ("group", (KnowledgeDocument.scope_type == "group") & or_(*group_conditions)),
        ("genus", (KnowledgeDocument.scope_type == "genus") & KnowledgeDocument.scope_value.in_(genera)),
        ("family", (KnowledgeDocument.scope_type == "family") & KnowledgeDocument.scope_value.in_(families)),
        ("general", KnowledgeDocument.scope_type == "general"),
    ]

    results, seen_ids = [], set()

    # Thu thập candidate theo taxonomy fallback
    for match_level, condition in levels:
        if len(results) >= RERANK_CANDIDATE_K:
            break

        rows = search_level(db=db, query_embedding=query_embedding, condition=condition, sections=sections, limit=RERANK_CANDIDATE_K - len(results), match_level=match_level)

        for result in rows:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                results.append(result)

    if use_reranker:
        results = rerank_results(query_text, results)

    results = select_top_results(results, detected_sections, top_k)
    return build_retrieval_response(results, candidates, detected_sections)