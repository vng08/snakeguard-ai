import json
from functools import lru_cache

from openai import OpenAI
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.models import SnakeSpecies


SYSTEM_INSTRUCTION = """
Bạn là SnakeGuard AI, trợ lý chuyên cung cấp thông tin về các loài rắn và an toàn khi gặp rắn.

Nhiệm vụ của bạn là đánh giá liệu CONTEXT có đủ căn cứ để trả lời câu hỏi hay không.

Quy tắc:
- Chỉ sử dụng thông tin trong CONTEXT; không tự bổ sung, suy đoán hoặc dùng kiến thức bên ngoài.
- Chỉ coi là đủ thông tin khi CONTEXT trực tiếp hỗ trợ câu trả lời.
- Thông tin ở scope genus, family hoặc group có thể được sử dụng nếu nội dung rõ ràng áp dụng cho nhóm chứa loài đang được hỏi.
- Không áp dụng thông tin của một species khác cho species đang được hỏi.
- Nếu CONTEXT không đủ, không cố tạo câu trả lời.
- Nếu đủ, trả lời tự nhiên, rõ ràng, trực tiếp và mặc định bằng tiếng Việt.
- Nếu biết tên tiếng Việt và tên khoa học, có thể giới thiệu tự nhiên như "Rắn cạp nong (Bungarus fasciatus)".
- Khi có nhiều đoạn liên quan, tổng hợp thành câu trả lời mạch lạc thay vì sao chép từng đoạn.
- Không nhắc đến CONTEXT hoặc quá trình truy xuất.

Với câu hỏi về rắn cắn:
- Ưu tiên thông tin y khoa.
- Không áp dụng thông tin của loài hoặc nhóm khác nếu không có căn cứ.
- Nếu là tình huống rắn cắn thực tế và CONTEXT có hướng dẫn phù hợp, khuyến nghị người dùng đến cơ sở y tế càng sớm càng tốt.
- Không đưa thêm hướng dẫn y tế ngoài CONTEXT.

Bạn PHẢI trả về JSON hợp lệ theo đúng cấu trúc:
{
  "can_answer": true hoặc false,
  "answer": "câu trả lời" hoặc null
}

Nếu CONTEXT không đủ:
{
  "can_answer": false,
  "answer": null
}
"""

CASUAL_SYSTEM_INSTRUCTION = """
Bạn là SnakeGuard AI.

Hãy trò chuyện tự nhiên, thân thiện và ngắn gọn với người dùng bằng tiếng Việt.
Đây là hội thoại thông thường nên không cần sử dụng cơ sở kiến thức về rắn.
Không tự nhận rằng bạn biết thông tin thời gian thực nếu chưa được cung cấp.
"""

SECTION_LABELS = {
    "diagnosis": "Chẩn đoán, triệu chứng, dấu hiệu",
    "treatment": "Điều trị",
    "first_aid": "Sơ cứu",
    "identification": "Đặc điểm nhận dạng",
    "distribution": "Phân bố",
    "habitat_behavior": "Môi trường sống, tập tính",
    "venom_danger": "Nọc độc, mức độ nguy hiểm",
}


@lru_cache(maxsize=1)
def get_llm_client():
    """Khởi tạo Groq client một lần."""
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY chưa được cấu hình.")

    return OpenAI(
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )


def load_species_metadata(db: Session, results):
    """Lấy metadata loài liên quan đến retrieved chunks."""
    species_ids = {result["species_id"] for result in results if result.get("species_id")}
    species_names = {
        result["scope_value"]
        for result in results
        if result.get("scope_type") == "species" and result.get("scope_value")
    }

    if not species_ids and not species_names:
        return {}, {}

    conditions = []

    if species_ids:
        conditions.append(SnakeSpecies.id.in_(species_ids))

    if species_names:
        conditions.append(SnakeSpecies.binomial_name.in_(species_names))

    species_list = db.execute(
        select(SnakeSpecies).where(or_(*conditions))
    ).scalars().all()

    by_id = {species.id: species for species in species_list}
    by_name = {species.binomial_name: species for species in species_list}

    return by_id, by_name


def build_context(db: Session, results):
    """Chuyển retrieval results thành context cho LLM."""
    chunks = []
    species_by_id, species_by_name = load_species_metadata(db, results)

    for index, result in enumerate(results, 1):
        scope = result["scope_value"] or "Rắn cắn nói chung"
        section = SECTION_LABELS.get(result["section"], result["section"])
        species = species_by_id.get(result.get("species_id")) or species_by_name.get(scope)
        parts = [f"[{index}]"]

        if species:
            parts.extend([
                f"Species: {species.binomial_name}",
                f"Tên tiếng Việt: {species.vietnamese_name or 'Không có dữ liệu'}",
                f"Genus: {species.genus or 'Không có dữ liệu'}",
                f"Family: {species.family or 'Không có dữ liệu'}",
            ])
        else:
            parts.append(f"Scope: {scope}")

        parts.append(f"Section: {section}")

        if result.get("subsection"):
            parts.append(f"Subsection: {result['subsection']}")

        parts.extend([
            f"Source: {result['source']}",
            f"Content: {result['content']}",
        ])

        chunks.append("\n".join(parts))

    return "\n\n".join(chunks)


def build_prompt(db: Session, question, results):
    """Tạo prompt gồm câu hỏi và retrieved context."""
    context = build_context(db, results)

    return f"""
CÂU HỎI CỦA NGƯỜI DÙNG:
{question}

CONTEXT:
{context}

Hãy đánh giá CONTEXT có đủ để trả lời câu hỏi hay không và trả về JSON theo đúng cấu trúc đã yêu cầu.
""".strip()


def parse_knowledge_answer(content):
    """Parse JSON từ LLM."""
    content = content.strip()

    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```").strip()
        content = content.removesuffix("```").strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Groq trả về JSON không hợp lệ: {content}") from exc

    can_answer = data.get("can_answer")
    answer = data.get("answer")

    if not isinstance(can_answer, bool):
        raise RuntimeError("Groq không trả về can_answer hợp lệ.")

    if can_answer and not answer:
        raise RuntimeError("Groq báo can_answer=true nhưng không có answer.")

    return {
        "can_answer": can_answer,
        "answer": answer.strip() if isinstance(answer, str) else None,
    }


def generate_answer(db: Session, question, results):
    """Sinh câu trả lời grounded và đánh giá context có đủ không."""
    if not results:
        return {
            "can_answer": False,
            "answer": None,
        }

    client = get_llm_client()
    prompt = build_prompt(db, question, results)

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1
    )

    content = (response.choices[0].message.content or "").strip()

    if not content:
        raise RuntimeError("Groq không trả về nội dung.")

    return parse_knowledge_answer(content)


def generate_casual_answer(message: str):
    """Trả lời hội thoại thông thường không qua RAG."""
    client = get_llm_client()

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": CASUAL_SYSTEM_INSTRUCTION},
            {"role": "user", "content": message},
        ],
        temperature=0.5,
        max_completion_tokens=300,
    )

    answer = (response.choices[0].message.content or "").strip()

    if not answer:
        raise RuntimeError("Groq không trả về nội dung.")

    return answer