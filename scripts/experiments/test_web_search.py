import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from backend.app.services.rag.tools.web_search import search_web


def print_results(title: str, results: list[dict]):
    """In kết quả Tavily để kiểm tra nội dung và nguồn."""
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)

    for index, item in enumerate(results, start=1):
        print(
            f"\n[{index}] score={item.get('web_score')} "
            f"| source={item['source']} | title={item['title']}"
        )
        print(f"URL: {item['source_url']}")
        print(item["content"][:500])


def test_fast_web_search():
    """Test Tavily basic search trong Fast mode."""
    state = {
        "session_id": "test-fast",
        "message": "Rắn hổ đất cắn thì nên xử lý như thế nào?",
        "history": [],
        "search_mode": "fast",
        "analysis": {
            "route": "species",
            "species_text": "Rắn hổ đất",
            "standalone_query": "Rắn hổ đất cắn thì nên xử lý như thế nào?",
        },
        "evaluation": {
            "sufficient": False,
            "confidence": 0.9,
            "missing_information": "Cần thêm thông tin sơ cứu và điều trị đáng tin cậy.",
        },
    }

    result = search_web(state)
    print_results("FAST WEB SEARCH", result["web_documents"])


def test_deep_web_search():
    """Test Tavily advanced search trong Deep mode."""
    state = {
        "session_id": "test-deep",
        "message": "Rắn hổ đất cắn thì nên xử lý như thế nào?",
        "history": [],
        "search_mode": "deep",
        "analysis": {
            "route": "species",
            "species_text": "Rắn hổ đất",
            "standalone_query": "Rắn hổ đất cắn thì nên xử lý như thế nào?",
        },
        "evaluation": {
            "sufficient": False,
            "confidence": 0.9,
            "missing_information": "Cần thêm thông tin về triệu chứng nguy hiểm và hướng điều trị.",
        },
    }

    result = search_web(state)
    print_results("DEEP WEB SEARCH", result["web_documents"])


def main():
    test_fast_web_search()
    test_deep_web_search()


if __name__ == "__main__":
    main()