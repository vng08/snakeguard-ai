from backend.app.core.config import settings
from backend.app.schemas.chat import QueryAnalysis
from backend.app.services.rag.llm_service import get_llm_client
from backend.app.services.rag.orchestration.state import ChatState


SYSTEM_PROMPT = """
Bạn là Query Analyzer của hệ thống SnakeGuard AI.

Nhiệm vụ:
1. Phân loại hướng xử lý của câu hỏi thành một trong ba route:
   - casual: hội thoại thông thường hoặc câu hỏi về khả năng/chức năng của trợ lý, không cần tra cứu kiến thức.
   - global: người dùng thực sự yêu cầu kiến thức về rắn nói chung nhưng không nói về một loài cụ thể.
   - species: câu hỏi đang nói về một loài hoặc tên rắn cụ thể.
2. Trích xuất tên rắn mà người dùng đang nhắc tới vào species_text.
3. Sử dụng lịch sử hội thoại để hiểu các tham chiếu như "nó", "loài đó", "con đầu tiên".
4. Viết lại câu hỏi thành standalone_query để câu hỏi có đầy đủ ngữ cảnh mà không cần đọc lịch sử.

Quy tắc:
- Không chọn global chỉ vì câu hỏi có chứa từ "rắn".
- Nếu người dùng chỉ đang hỏi trợ lý có thể giúp tìm, nhận dạng, giải thích hoặc cung cấp thông tin hay không thì chọn casual.
- Chỉ chọn global khi người dùng thực sự đang yêu cầu kiến thức về rắn nói chung.
- Không tự suy đoán tên khoa học.
- Không tự sửa tên rắn mà người dùng cung cấp.
- Nếu route là casual hoặc global thì species_text phải là null.
- Nếu câu hỏi hiện tại phụ thuộc vào lịch sử, standalone_query phải bổ sung ngữ cảnh cần thiết từ lịch sử.
- Không thay đổi ý định của người dùng.
- Chỉ trả về JSON đúng cấu trúc được yêu cầu.

Ví dụ:
- "Bạn giúp tôi tìm loại rắn được chứ?" -> casual
- "Bạn có thể nhận dạng rắn không?" -> casual
- "Bạn có thể giúp tôi tìm hiểu về rắn không?" -> casual
- "Rắn thường hoạt động vào thời gian nào?" -> global
- "Bị rắn cắn thì nên sơ cứu thế nào?" -> global
- "Rắn hổ đất sống ở đâu?" -> species
"""


def analyze_query(state: ChatState) -> dict:
    """Phân tích câu hỏi hiện tại dựa trên toàn bộ lịch sử hội thoại."""
    client = get_llm_client()
    history_text = _format_history(state.get("history", []))

    prompt = f"""
Lịch sử hội thoại:
{history_text}

Câu hỏi hiện tại:
{state["message"]}

Trả về JSON theo cấu trúc:
{{
    "route": "casual | global | species",
    "species_text": "tên rắn được nhắc tới" hoặc null,
    "standalone_query": "câu hỏi đã được viết lại đầy đủ ngữ cảnh"
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

    analysis = QueryAnalysis.model_validate_json(response.choices[0].message.content)
    return {"analysis": analysis.model_dump()}


def _format_history(history: list) -> str:
    """Chuyển toàn bộ lịch sử hội thoại thành text cho Query Analyzer."""
    if not history:
        return "Không có lịch sử hội thoại."

    lines = []
    for message in history:
        role = message.role if hasattr(message, "role") else message.get("role")
        content = message.content if hasattr(message, "content") else message.get("content")
        lines.append(f"{role}: {content}")

    return "\n".join(lines)