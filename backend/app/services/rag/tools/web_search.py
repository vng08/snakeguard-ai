from urllib.parse import urlparse
from tavily import TavilyClient
from backend.app.core.config import settings
from backend.app.services.rag.orchestration.state import ChatState


_tavily_client = None


def get_tavily_client() -> TavilyClient:
    """Lazy-load Tavily client và dùng chung trong toàn bộ Agentic RAG."""
    global _tavily_client

    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)

    return _tavily_client


def search_web(state: ChatState) -> dict:
    """Tìm thêm thông tin trên web khi knowledge base chưa đủ."""
    query = state["analysis"]["standalone_query"]
    missing_information = state.get("evaluation", {}).get("missing_information")
    search_query = _build_search_query(query, missing_information)

    client = get_tavily_client()
    search_depth = "basic" if state["search_mode"] == "fast" else "advanced"
    response = client.search(query=search_query, search_depth=search_depth, max_results=5, include_answer=False, include_raw_content=False)

    documents = [_build_document(result) for result in response.get("results", [])]
    return {"web_documents": documents}


def _build_search_query(query: str, missing_information: str | None) -> str:
    """Bổ sung phần thông tin còn thiếu vào web search query."""
    if not missing_information:
        return query

    return f"{query}\nThông tin cần tìm thêm: {missing_information}"


def _build_document(result: dict) -> dict:
    """Chuẩn hoá Tavily result thành web document dùng trong RAG."""
    url = result.get("url", "")

    return {
        "title": result.get("title", ""),
        "content": result.get("content", ""),
        "source": _get_domain(url),
        "source_url": url,
        "web_score": result.get("score"),
    }


def _get_domain(url: str) -> str:
    """Lấy domain để hiển thị nguồn web ngắn gọn."""
    return urlparse(url).netloc or "Web"