import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from backend.app.db.session import SessionLocal
from backend.app.services.retrieval_service import search_knowledge


TEST_QUERIES = [
    "Rắn cạp nia cắn có triệu chứng gì?",
    "Rắn cạp nong có đặc điểm nhận dạng như thế nào?",
    "Bị rắn cắn nên sơ cứu như thế nào?",
    "Rắn lục đầu bạc có đặc điểm gì?",
]


def print_results(query, retrieval):
    """In kết quả retrieval."""
    results = retrieval["results"]

    print(f"\n{'=' * 80}")
    print(f"QUERY: {query}")
    print(f"Resolved species: {retrieval['resolved_species']}")
    print(f"Detected sections: {retrieval['detected_sections']}")
    print(f"Ambiguous: {retrieval['is_ambiguous']}")
    print(f"Candidate species: {retrieval['candidate_species']}")

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] score={result['score']:.4f} | {result['scope_value']} | {result['section']} | {result['subsection']} | match={result['match_level']}")
        print(result["content"][:500])


def main():
    db = SessionLocal()

    try:
        for query in TEST_QUERIES:
            retrieval = search_knowledge(db, query, top_k=5)
            print_results(query, retrieval)
    finally:
        db.close()


if __name__ == "__main__":
    main()