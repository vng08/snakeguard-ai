from backend.app.core.config import settings
from backend.app.services.llm_service import get_llm_client


ROUTER_SYSTEM_INSTRUCTION = """
Phân loại tin nhắn của người dùng thành đúng một trong hai nhãn:

snake:
- Câu hỏi hoặc nội dung liên quan đến rắn.
- Bao gồm loài rắn, nhận dạng, phân bố, môi trường sống, tập tính, độc tính.
- Bao gồm rắn cắn, triệu chứng, sơ cứu, điều trị và an toàn khi gặp rắn.

casual:
- Chào hỏi, trò chuyện thông thường hoặc chủ đề không liên quan đến rắn.

Chỉ trả về đúng một từ: snake hoặc casual.
"""


def classify_message(message: str):
    """Phân loại câu hỏi liên quan đến rắn hay hội thoại thông thường."""
    client = get_llm_client()

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_INSTRUCTION},
            {"role": "user", "content": message},
        ],
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=200,
    )

    raw = response.choices[0].message.content or ""
    route = raw.strip().lower()

    print(f"ROUTER RAW: {raw!r}")

    if route not in {"snake", "casual"}:
        return "snake"

    return route