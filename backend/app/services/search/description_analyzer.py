from backend.app.schemas.search import DescriptionAnalysis
from backend.app.services.rag.llm_service import generate_json


DESCRIPTION_SYSTEM_INSTRUCTION = """
Bạn là Description Analyzer của SnakeGuard AI.

Phân tích mô tả của người dùng để phục vụ tìm kiếm loài rắn.

route:
- snake: mô tả chứa đặc điểm có thể dùng để nhận dạng hoặc tìm loài rắn.
- unspecified: nội dung không liên quan đến rắn hoặc không cung cấp đặc điểm hữu ích.

focus:
- visual: chủ yếu là đặc điểm nhìn thấy như màu sắc, hoa văn, hình dạng đầu, thân hoặc kích thước.
- contextual: chủ yếu là môi trường sống, tập tính, thời gian hoạt động, vị trí bắt gặp hoặc phân bố.
- mixed: có cả đặc điểm visual và contextual đáng kể.

Quy tắc:
- Nếu route là unspecified thì focus phải là mixed.
- Không suy đoán loài rắn.
- Chỉ trả về JSON đúng cấu trúc:
{
  "route": "snake | unspecified",
  "focus": "visual | mixed | contextual"
}
"""


def analyze_description(description: str) -> DescriptionAnalysis:
    """Phân tích description và xác định hướng hybrid search."""
    try:
        result = generate_json([
            {"role": "system", "content": DESCRIPTION_SYSTEM_INSTRUCTION},
            {"role": "user", "content": description},
        ], temperature=0)

        return DescriptionAnalysis.model_validate(result)

    except (ValueError, TypeError):
        return DescriptionAnalysis(route="unspecified", focus="mixed")