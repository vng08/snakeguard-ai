from backend.app.core.config import settings
from backend.app.services.llm_service import get_llm_client


DESCRIPTION_SYSTEM_INSTRUCTION = """
Phân loại mô tả của người dùng thành đúng một trong hai nhãn:

snake:
- Nội dung mô tả các đặc điểm có thể dùng để nhận dạng rắn.
- Ví dụ: màu sắc, hoa văn, hình dạng đầu, hình dạng thân, kích thước,
  môi trường sống, tập tính hoặc vị trí bắt gặp.

unspecified:
- Nội dung không liên quan đến việc mô tả hoặc nhận dạng rắn.
- Hoặc nội dung không cung cấp đặc điểm hữu ích để tìm loài rắn.

Chỉ trả về đúng một từ: snake hoặc unspecified.
"""


def classify_description(description: str):
    client = get_llm_client()

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": DESCRIPTION_SYSTEM_INSTRUCTION},
            {"role": "user", "content": description},
        ],
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=200,
    )

    raw = response.choices[0].message.content or ""
    route = raw.strip().lower()

    print(f"DESCRIPTION CLASSIFIER RAW: {raw!r}")

    if route not in {"snake", "unspecified"}:
        return "unspecified"

    return route