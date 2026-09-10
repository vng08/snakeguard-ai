from backend.app.schemas.chat import AnswerEvaluation
from backend.app.services.rag.llm_service import generate_json
from backend.app.services.rag.orchestration.state import ChatState

SYSTEM_PROMPT = """
Bạn là Answer Evaluator của SnakeGuard AI.

Nhiệm vụ:
Đánh giá context hiện có có đủ để trả lời đúng câu hỏi của người dùng hay không.

Quy tắc:
- Chỉ đánh giá dựa trên context được cung cấp, không dùng kiến thức bên ngoài.
- Đánh giá theo đúng phạm vi câu hỏi, không yêu cầu context phải chứa toàn bộ thông tin về một loài.
- sufficient=true nếu context có đủ thông tin trực tiếp và đáng tin cậy để trả lời điều người dùng đang hỏi.
- Nếu câu hỏi chỉ hỏi một thuộc tính như môi trường sống, độc tính, kích thước hoặc tập tính, chỉ cần context đủ cho thuộc tính đó.
- Không đánh giá thiếu chỉ vì context không có các thông tin mà người dùng không hỏi.
- Với câu hỏi rộng như "cho tôi thông tin về loài X", sufficient=true nếu context có đủ thông tin hữu ích để tạo một câu trả lời có giới hạn; không cần đầy đủ mọi đặc điểm của loài.
- sufficient=false nếu context không trả lời được câu hỏi, thiếu thông tin cần thiết hoặc có mâu thuẫn quan trọng.
- Với câu hỏi y tế hoặc an toàn, nếu context mâu thuẫn hoặc không đủ rõ ràng thì sufficient=false.
- Với câu hỏi về một loài cụ thể, context phải liên quan đúng loài hoặc taxonomy phù hợp.
- confidence là mức độ chắc chắn của quyết định, từ 0 đến 1.
- Nếu sufficient=true thì missing_information=null.
- Nếu sufficient=false thì missing_information chỉ mô tả ngắn gọn thông tin thực sự còn thiếu để trả lời câu hỏi hiện tại.
- Không trả lời câu hỏi của người dùng.
- Chỉ trả về JSON đúng cấu trúc yêu cầu.

Ví dụ:
Câu hỏi: "Rắn cạp nia Nam sống ở đâu?"
Context có thông tin về môi trường sống và phân bố, nhưng không có kích thước hoặc thức ăn.
→ sufficient=true

Câu hỏi: "Cho tôi thông tin về rắn cạp nia Nam."
Context có tên khoa học, phân bố, môi trường sống và độc tính.
→ sufficient=true

Câu hỏi: "Rắn cạp nia Nam ăn gì?"
Context chỉ có phân bố và độc tính.
→ sufficient=false
→ missing_information="Thiếu thông tin về chế độ ăn của loài."
"""


def evaluate_answer(state: ChatState) -> dict:
    """Đánh giá context hiện tại có đủ để trả lời câu hỏi hay không."""
    query = state["analysis"]["standalone_query"]
    documents = state.get("kb_documents", []) + state.get("web_documents", [])

    if not documents:
        evaluation = AnswerEvaluation(
            sufficient=False,
            confidence=1.0,
            missing_information="Không có context phù hợp để trả lời câu hỏi.",
        )
        return {"evaluation": evaluation.model_dump()}

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

    result = generate_json([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ], temperature=0)

    evaluation = AnswerEvaluation.model_validate(result)
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