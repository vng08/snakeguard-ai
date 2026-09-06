from sqlalchemy.orm import Session

from backend.app.services.llm_service import generate_answer, generate_casual_answer
from backend.app.services.message_router_service import classify_message
from backend.app.services.retrieval_service import search_knowledge
from backend.app.services.web_search_service import WebSearchUnavailableError, search_web


CHAT_TOP_K = 5

SPECIES_SPECIFIC_SECTIONS = {"identification", "distribution", "habitat_behavior", "venom_danger"}
MEDICAL_SECTIONS = {"diagnosis", "treatment", "first_aid"}
SHARED_MEDICAL_SCOPES = {"genus", "family", "group", "general"}


def build_sources(results):
    """Chuẩn hóa và loại bỏ source KB trùng nhau."""
    sources, seen = [], set()

    for result in results:
        key = (result["source"], result.get("source_url"), result["section"], result.get("scope_value"))

        if key in seen:
            continue

        seen.add(key)
        sources.append({"source": result["source"], "source_url": result.get("source_url"), "section": result["section"], "scope_value": result.get("scope_value")})

    return sources


def ambiguity_needs_clarification(retrieval):
    """Kiểm tra ambiguity có thực sự cần hỏi lại user không."""
    if not retrieval["is_ambiguous"]:
        return False

    sections = set(retrieval["detected_sections"])

    if not sections:
        return True

    # Medical có thể dùng knowledge ở scope chung
    if sections.issubset(MEDICAL_SECTIONS):
        return not any(result["scope_type"] in SHARED_MEDICAL_SCOPES for result in retrieval["results"])

    return bool(sections & SPECIES_SPECIFIC_SECTIONS)


def build_ambiguous_answer(candidate_species):
    """Phản hồi khi tên gọi có thể chỉ nhiều species."""
    names = ", ".join(candidate_species)
    return f"Tên loài bạn hỏi có thể tương ứng với nhiều loài: {names}. Bạn có thể cung cấp thêm tên đầy đủ, vị trí gặp rắn hoặc ảnh để mình xác định chính xác hơn được không?"


def fallback_to_web(message, retrieval):
    """Tìm web khi KB không đủ thông tin."""
    try:
        web_result = search_web(message)
        answer = (
            "Tôi đã kiểm tra cơ sở kiến thức hiện có nhưng chưa tìm thấy thông tin phù hợp "
            "để trả lời đầy đủ câu hỏi của bạn. Vì vậy, tôi đã tìm thêm thông tin trên Internet."
            f"\n\n{web_result['answer']}"
        )

        return {
            "answer": answer,
            "source_type": "web",
            "sources": web_result["sources"],
            "is_ambiguous": retrieval["is_ambiguous"],
            "needs_clarification": False,
            "candidate_species": retrieval["candidate_species"],
            "notice": "Tôi đã tìm thêm thông tin trên Internet.",
        }

    except WebSearchUnavailableError:
        return {
            "answer": "Cơ sở kiến thức hiện tại chưa đủ để trả lời câu hỏi này và tính năng tìm kiếm Internet đang tạm thời không khả dụng. Bạn có thể thử lại sau.",
            "source_type": "unavailable",
            "sources": [],
            "is_ambiguous": retrieval["is_ambiguous"],
            "needs_clarification": False,
            "candidate_species": retrieval["candidate_species"],
            "notice": "Tìm kiếm Internet hiện không khả dụng.",
        }


def chat(db: Session, message: str):
    """Điều phối casual chat, KB retrieval, clarification và web fallback."""

    # Ngoài luồng rắn -> trò chuyện bình thường
    if classify_message(message) == "casual":
        return {
            "answer": generate_casual_answer(message),
            "source_type": "casual",
            "sources": [],
            "is_ambiguous": False,
            "needs_clarification": False,
            "candidate_species": [],
            "notice": None,
        }

    retrieval = search_knowledge(db, message, top_k=CHAT_TOP_K, use_reranker=True)
    results = retrieval["results"]

    # Ambiguous và cần xác định chính xác species
    if ambiguity_needs_clarification(retrieval):
        return {
            "answer": build_ambiguous_answer(retrieval["candidate_species"]),
            "source_type": "knowledge_base",
            "sources": [],
            "is_ambiguous": True,
            "needs_clarification": True,
            "candidate_species": retrieval["candidate_species"],
            "notice": None,
        }

    # Không retrieve được context
    if not results:
        return fallback_to_web(message, retrieval)

    # LLM đánh giá context và sinh câu trả lời
    kb_result = generate_answer(db, message, results)

    if not kb_result["can_answer"]:
        return fallback_to_web(message, retrieval)

    return {
        "answer": kb_result["answer"],
        "source_type": "knowledge_base",
        "sources": build_sources(results),
        "is_ambiguous": retrieval["is_ambiguous"],
        "needs_clarification": False,
        "candidate_species": retrieval["candidate_species"],
        "notice": None,
    }