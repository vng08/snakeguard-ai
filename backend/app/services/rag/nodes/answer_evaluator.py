from backend.app.core.config import settings
from backend.app.schemas.chat import AnswerEvaluation
from backend.app.services.rag.llm_service import get_llm_client
from backend.app.services.rag.orchestration.state import ChatState


SYSTEM_PROMPT = """
Bạn là Answer Evaluator của hệ thống SnakeGuard AI.

Nhiệm vụ của bạn là đánh giá liệu context được cung cấp có đủ để trả lời câu hỏi của người dùng một cách chính xác hay chưa.

Quy tắc:
- Chỉ đánh giá dựa trên context được cung cấp, không sử dụng kiến thức bên ngoài.
- sufficient=true chỉ khi context chứa đủ thông tin trực tiếp và liên quan để trả lời câu hỏi.
- Context chỉ liên quan đến chủ đề nhưng không trả lời đúng điều người dùng hỏi thì phải đánh giá là chưa đủ.
- Nếu context có thông tin mâu thuẫn quan trọng, đặc biệt với câu hỏi y khoa hoặc an toàn, phải đánh giá là chưa đủ.
- Nếu câu hỏi yêu cầu thông tin về một loài cụ thể, context phải có thông tin phù hợp với loài hoặc taxonomy liên quan.
- confidence thể hiện mức độ chắc chắn của bạn về quyết định đánh giá, từ 0 đến 1.
- Nếu sufficient=true thì missing_information phải là null.
- Nếu sufficient=false thì missing_information phải mô tả ngắn gọn thông tin còn thiếu hoặc vấn đề khiến context chưa đủ.
- Không trả lời câu hỏi của người dùng.
- Chỉ trả về JSON đúng cấu trúc được yêu cầu.
"""


def evaluate_answer(state: ChatState) -> dict:
    """Đánh giá context hiện tại có đủ để trả lời câu hỏi hay không."""
    query = state["analysis"]["standalone_query"]
    documents = state.get("kb_documents", []) + state.get("web_documents", [])

    if not documents:
        evaluation = AnswerEvaluation(sufficient=False, confidence=1.0, missing_information="Không có context phù hợp để trả lời câu hỏi.")
        return {"evaluation": evaluation.model_dump()}

    client = get_llm_client()
    context = _format_documents(documents)

    prompt = f"""
Câu hỏi:
{query}

Context:
{context}

Trả về JSON theo cấu trúc:
{{
    "sufficient": true hoặc false,
    "confidence": số từ 0 đến 1,
    "missing_information": "thông tin còn thiếu" hoặc null
}}
"""

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    evaluation = AnswerEvaluation.model_validate_json(response.choices[0].message.content)
    return {"evaluation": evaluation.model_dump()}


def _format_documents(documents: list[dict]) -> str:
    """Chuyển retrieved documents thành context cho evaluator."""
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