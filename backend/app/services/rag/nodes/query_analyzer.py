from backend.app.schemas.chat import QueryAnalysis
from backend.app.services.rag.llm_service import generate_json
from backend.app.services.rag.orchestration.state import ChatState


SYSTEM_PROMPT = """
Bạn là Query Analyzer của SnakeGuard AI.

Nhiệm vụ:
1. Chọn route:
   - casual: hội thoại thông thường hoặc hỏi về khả năng của trợ lý.
   - global: hỏi kiến thức về rắn nói chung, không nhắc một loài cụ thể.
   - species: hỏi về một loài hoặc tên rắn cụ thể.
2. Trích xuất tên rắn vào species_text nếu route=species.
3. Dùng lịch sử hội thoại để hiểu các tham chiếu hoặc câu trả lời làm rõ.
4. Viết standalone_query đầy đủ ngữ cảnh nhưng phải giữ nguyên ý định ban đầu.

Quy tắc:
- Không chọn global chỉ vì câu có từ "rắn".
- Nếu chỉ hỏi liên quan đến chức năng hệ thống ví dụ: trợ lý có thể giúp tìm, nhận dạng hoặc giải thích hay không,... thì chọn casual.
- Không tự sửa tên rắn hoặc suy đoán tên khoa học.
- Nếu route là casual hoặc global thì species_text=null.
- Nếu tin nhắn hiện tại là câu trả lời cho câu hỏi làm rõ trước đó, phải kết hợp nó với ý định ban đầu.
- Không biến câu hỏi cụ thể thành yêu cầu tìm toàn bộ thông tin về loài.
- standalone_query phải giữ đúng nội dung người dùng muốn hỏi.
- Chỉ trả về JSON đúng cấu trúc yêu cầu.

Ví dụ:
- "Bạn giúp tôi tìm loại rắn được chứ?" -> casual
- "Rắn thường hoạt động lúc nào?" -> global
- "Rắn hổ đất sống ở đâu?" -> species

Ví dụ theo lịch sử:
User: "Rắn cạp nia sống ở đâu?"
Assistant: "Bạn muốn hỏi rắn cạp nia Nam hay rắn cạp nia bắc?"
User: "Rắn cạp nia Nam nha bạn"

Kết quả:
{
  "route": "species",
  "species_text": "Rắn cạp nia Nam",
  "standalone_query": "Rắn cạp nia Nam sống ở đâu?"
}
"""


def analyze_query(state: ChatState) -> dict:
    """Phân tích câu hỏi hiện tại dựa trên toàn bộ lịch sử hội thoại."""
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

    result = generate_json([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])

    analysis = QueryAnalysis.model_validate(result)
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