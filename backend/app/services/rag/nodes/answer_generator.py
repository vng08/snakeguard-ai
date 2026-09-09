from backend.app.core.config import settings
from backend.app.services.rag.llm_service import get_llm_client
from backend.app.services.rag.orchestration.state import ChatState


GROUNDED_SYSTEM_PROMPT = """
Bạn là trợ lý SnakeGuard AI chuyên hỗ trợ kiến thức về rắn.

Nhiệm vụ:
- Trả lời câu hỏi dựa trên context được cung cấp.
- Ưu tiên thông tin trực tiếp liên quan đến câu hỏi.
- Không thêm thông tin không có trong context.
- Nếu context chứa cả kiến thức chung và kiến thức theo loài, hãy kết hợp chúng khi phù hợp.
- Với nội dung y tế hoặc an toàn, trình bày rõ ràng, thận trọng và không suy diễn vượt quá nguồn.
- Không bịa tên loài, đặc điểm, độc tính, cách điều trị hoặc nguồn tham khảo.
- Không tự dịch, đổi tên hoặc gán tên phổ biến khác cho loài nếu context không cung cấp.
- Trả lời tự nhiên bằng ngôn ngữ của người dùng.

Quy tắc về Species Resolver:
- Nếu tên rắn mà người dùng nhập khớp chính xác, sử dụng đúng tên đã resolve.
- Nếu tên rắn mà người dùng nhập không khớp chính xác và Species Resolver chỉ tìm được một loài gần giống, phải thừa nhận rõ điều đó trước khi trả lời.
- Phải nói tên rắn ban đầu mà người dùng nhập và loài mà hệ thống suy đoán.
- Không được trình bày loài suy đoán như thể người dùng đã nhập đúng tên đó.
- Khi nhắc đến loài đã resolve, ưu tiên matched_name và binomial_name do Species Resolver cung cấp.

Quy tắc về Web Search:
- Nếu hệ thống đã tìm thêm thông tin trên Internet, phải nói rõ với người dùng rằng câu trả lời có sử dụng thêm thông tin tìm kiếm từ Internet.
- Không được khiến người dùng hiểu rằng toàn bộ thông tin chỉ đến từ knowledge base nội bộ khi Web Search đã được sử dụng.
"""


CASUAL_SYSTEM_PROMPT = """
Bạn là trợ lý SnakeGuard AI.

Thông tin về hệ thống:
- SnakeGuard AI hiện hỗ trợ dữ liệu của khoảng 109 loài rắn.
- Hệ thống có thể nhận diện rắn từ hình ảnh.
- Hệ thống có thể tìm loài rắn dựa trên mô tả về màu sắc, hoa văn, hình dạng, môi trường sống, tập tính hoặc vị trí bắt gặp.
- Chatbot có thể hỏi đáp kiến thức về rắn bằng Agentic RAG.
- Kiến thức về rắn được lấy chủ yếu từ knowledge base nội bộ; khi thông tin chưa đủ, hệ thống có thể tìm thêm từ Internet.
- Chatbot hỗ trợ hai chế độ tìm kiếm Fast và Deep.

Quy tắc hội thoại:
- Hãy trò chuyện tự nhiên và thân thiện bằng ngôn ngữ của người dùng.
- Có thể trả lời các câu hỏi đời thường như chào hỏi, học tập, công việc, ăn uống, thời tiết nói chung và các chủ đề thông thường khác.
- Không bắt buộc mọi câu hỏi phải liên quan đến rắn.
- Nếu người dùng hỏi về khả năng hoặc chức năng của SnakeGuard AI, hãy sử dụng thông tin hệ thống ở trên.
- Không tự bịa thêm chức năng hoặc thông tin về hệ thống.
- Với thông tin cần dữ liệu thời gian thực như thời tiết hiện tại, tin tức mới nhất, giá cả hoặc lịch trình hiện tại, không tự bịa nếu không có dữ liệu để xác minh.
- Không cần tra cứu knowledge base về rắn nếu câu hỏi chỉ là hội thoại thông thường.

Hãy trả lời dựa trên câu hỏi hiện tại và lịch sử hội thoại khi cần.
"""


INSUFFICIENT_SYSTEM_PROMPT = """
Bạn là trợ lý SnakeGuard AI.

Thông tin đã được tìm kiếm nhưng vẫn chưa đủ để xác minh câu trả lời một cách đáng tin cậy.

Quy tắc:
- Không tự suy đoán hoặc bịa thêm thông tin.
- Không khẳng định một loài không tồn tại chỉ vì chưa tìm thấy thông tin.
- Nói rõ rằng hiện chưa có đủ thông tin đáng tin cậy để xác minh.
- Nếu tên rắn người dùng nhập không được resolve chính xác, phải nói rõ điều đó.
- Nếu hệ thống đã tìm kiếm trên Internet, phải nói rõ rằng đã thử tìm thêm thông tin trên Internet nhưng vẫn chưa đủ để xác minh.
- Nếu có thông tin còn thiếu được cung cấp, giải thích ngắn gọn phần đó.
- Trả lời tự nhiên bằng ngôn ngữ của người dùng.
"""


def generate_answer(state: ChatState) -> dict:
    """Tạo câu trả lời cuối cùng dựa trên route và context hiện có."""
    route = state["analysis"]["route"]

    if route == "casual":
        answer = _generate_casual_answer(state)
        return {"answer": answer, "sources": []}

    evaluation = state.get("evaluation", {})

    if not evaluation.get("sufficient", False):
        answer = _generate_insufficient_answer(state)
        return {"answer": answer, "sources": _build_sources(state)}

    answer = _generate_grounded_answer(state)
    return {"answer": answer, "sources": _build_sources(state)}


def _generate_grounded_answer(state: ChatState) -> str:
    """Tạo câu trả lời dựa trên knowledge base và web context."""
    client = get_llm_client()
    query = state["analysis"]["standalone_query"]
    documents = state.get("kb_documents", []) + state.get("web_documents", [])
    context = _format_documents(documents)
    species_context = _format_species_context(state)
    search_context = _format_search_context(state)

    prompt = f"""
Câu hỏi:
{query}

Kết quả Species Resolver:
{species_context}

Trạng thái tìm kiếm:
{search_context}

Context:
{context}

Hãy trả lời câu hỏi dựa trên context trên và tuân thủ các quy tắc về Species Resolver và Web Search.
"""

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def _generate_casual_answer(state: ChatState) -> str:
    """Tạo câu trả lời cho hội thoại không cần retrieval."""
    client = get_llm_client()
    history = _format_history(state.get("history", []))

    prompt = f"""
Lịch sử hội thoại:
{history}

Tin nhắn hiện tại:
{state["message"]}
"""

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": CASUAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content.strip()


def _generate_insufficient_answer(state: ChatState) -> str:
    """Tạo câu trả lời an toàn khi KB và web vẫn chưa đủ thông tin."""
    client = get_llm_client()
    query = state["analysis"]["standalone_query"]
    missing_information = state.get("evaluation", {}).get("missing_information")
    species_context = _format_species_context(state)
    search_context = _format_search_context(state)

    prompt = f"""
Câu hỏi:
{query}

Kết quả Species Resolver:
{species_context}

Trạng thái tìm kiếm:
{search_context}

Thông tin chưa thể xác minh:
{missing_information or "Không có đủ thông tin đáng tin cậy để trả lời."}
"""

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": INSUFFICIENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


def _format_species_context(state: ChatState) -> str:
    """Mô tả kết quả Species Resolver cho Answer Generator."""
    if state["analysis"]["route"] != "species":
        return "Câu hỏi không yêu cầu resolve một loài cụ thể."

    species_text = state["analysis"].get("species_text")
    candidates = state.get("species_candidates", [])

    if not candidates:
        return f'Không tìm thấy species phù hợp trong database với tên rắn "{species_text}".'

    candidate = candidates[0]

    if candidate["match_type"] == "exact":
        return f'Tên rắn "{species_text}" khớp chính xác với {candidate["matched_name"]} ({candidate["binomial_name"]}).'

    return (
        f'Người dùng nhập tên rắn "{species_text}", nhưng không có kết quả khớp chính xác. '
        f'Hệ thống suy đoán loài gần nhất là {candidate["matched_name"]} ({candidate["binomial_name"]}) '
        f'bằng phương pháp {candidate["match_type"]}. Phải nói rõ điều này trước khi trả lời.'
    )


def _format_search_context(state: ChatState) -> str:
    """Cho Answer Generator biết hệ thống có sử dụng Web Search hay không."""
    if "web_documents" in state:
        return "Đã tìm thêm thông tin từ Internet bằng Web Search. Phải thông báo điều này cho người dùng."

    return "Không sử dụng Web Search; câu trả lời dựa trên knowledge base nội bộ."


def _format_documents(documents: list[dict]) -> str:
    """Chuyển KB và web documents thành context cho Answer Generator."""
    parts = []

    for index, document in enumerate(documents, start=1):
        source = document.get("source", "Unknown")
        section = document.get("section", "Unknown")
        content = document.get("content", "")

        parts.append(
            f"[Document {index}]\n"
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"Content: {content}"
        )

    return "\n\n".join(parts)


def _format_history(history: list) -> str:
    """Chuyển lịch sử hội thoại thành text cho casual answer."""
    if not history:
        return "Không có lịch sử hội thoại."

    lines = []

    for message in history:
        role = message.role if hasattr(message, "role") else message.get("role")
        content = message.content if hasattr(message, "content") else message.get("content")
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def _build_sources(state: ChatState) -> list[dict]:
    """Tạo danh sách nguồn đã dùng trong câu trả lời."""
    documents = state.get("kb_documents", []) + state.get("web_documents", [])
    sources = []
    seen = set()

    for document in documents:
        source = document.get("source")
        source_url = document.get("source_url")
        key = (source, source_url)

        if not source or key in seen:
            continue

        sources.append({"source": source, "source_url": source_url})
        seen.add(key)

    return sources