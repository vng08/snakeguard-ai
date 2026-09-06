from functools import lru_cache

from tavily import TavilyClient

from backend.app.core.config import settings
from backend.app.services.llm_service import get_llm_client


WEB_MAX_RESULTS = 4
WEB_MAX_CONTENT_CHARS = 1500

WEB_SYSTEM_INSTRUCTION = """
Bạn là SnakeGuard AI, trợ lý chuyên cung cấp thông tin về các loài rắn và an toàn khi gặp rắn.

Quy tắc:
- Chỉ sử dụng thông tin trong WEB CONTEXT được cung cấp; không tự bổ sung kiến thức bên ngoài.
- Ưu tiên thông tin từ nguồn chính thức, cơ quan y tế, tổ chức khoa học và nguồn đáng tin cậy.
- Bỏ qua thông tin không liên quan hoặc không đủ căn cứ.
- Nếu các nguồn không đủ hoặc mâu thuẫn, nói rõ sự không chắc chắn.
- Trả lời tự nhiên, rõ ràng, trực tiếp và mặc định bằng tiếng Việt.
- Có thể dẫn nguồn bằng [1], [2] tương ứng với WEB CONTEXT.
- Với thông tin y tế, không đưa ra chẩn đoán chắc chắn.
- Nếu người dùng mô tả trường hợp rắn cắn thực tế, khuyến nghị đến cơ sở y tế càng sớm càng tốt.

Trả lời ngắn gọn vừa đủ và ưu tiên tính chính xác.
"""


class WebSearchUnavailableError(Exception):
    """Web search tạm thời không khả dụng."""
    pass


@lru_cache(maxsize=1)
def get_tavily_client():
    """Khởi tạo Tavily client một lần."""
    if not settings.TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY chưa được cấu hình.")

    return TavilyClient(api_key=settings.TAVILY_API_KEY)


def build_web_context(results):
    """Chuyển Tavily results thành context ngắn cho Groq."""
    chunks = []

    for index, result in enumerate(results, 1):
        title = result.get("title") or "Không có tiêu đề"
        url = result.get("url") or ""
        content = (result.get("content") or "").strip()[:WEB_MAX_CONTENT_CHARS]

        chunks.append(
            f"[{index}]\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content}"
        )

    return "\n\n".join(chunks)


def build_web_sources(results):
    """Chuẩn hóa và loại bỏ source web trùng nhau."""
    sources = []
    seen_urls = set()

    for result in results:
        url = result.get("url")

        if not url or url in seen_urls:
            continue

        seen_urls.add(url)
        sources.append({
            "source": result.get("title") or url,
            "source_url": url,
        })

    return sources


def search_web(question: str):
    """Search bằng Tavily và tổng hợp câu trả lời bằng Groq."""
    tavily = get_tavily_client()

    try:
        response = tavily.search(
            query=question,
            search_depth="basic",
            max_results=WEB_MAX_RESULTS,
            include_answer=False,
            include_raw_content=False,
            include_images=False,
        )
    except Exception as exc:
        raise WebSearchUnavailableError("Không thể tìm kiếm thông tin trên Internet.") from exc

    results = response.get("results") or []

    if not results:
        raise WebSearchUnavailableError("Không tìm thấy kết quả web phù hợp.")

    context = build_web_context(results)
    client = get_llm_client()

    prompt = f"""
CÂU HỎI CỦA NGƯỜI DÙNG:
{question}

WEB CONTEXT:
{context}

Hãy trả lời câu hỏi chỉ dựa trên WEB CONTEXT ở trên.
""".strip()

    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": WEB_SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_completion_tokens=700,
        )
    except Exception as exc:
        raise WebSearchUnavailableError("Groq tạm thời không thể tổng hợp kết quả web.") from exc

    answer = (response.choices[0].message.content or "").strip()

    if not answer:
        raise WebSearchUnavailableError("Groq không trả về nội dung.")

    return {
        "answer": answer,
        "sources": build_web_sources(results),
    }