import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from backend.app.db.session import SessionLocal
from backend.app.services.search.description_analyzer import analyze_description
from backend.app.services.search.hybrid_retrieval_service import (
    hybrid_search,
    search_species_knowledge,
)
from backend.app.services.search.image_retrieval_service import search_by_description


def print_results(title, results, score_key):
    print(f"\n=== {title} ===")

    for i, result in enumerate(results, 1):
        print(
            f"{i}. {result['binomial_name']} | "
            f"{result['vietnamese_name']} | "
            f"score={result[score_key]:.4f}"
        )
        print(f"   image: {result.get('image_url')}")
        print(f"   MIVS: {result.get('is_mivs')}")


def main():
    db = SessionLocal()
    query = "rắn có lưng có các đốm sậm màu nối liền với đường sống lưng so le nhau"

    try:
        analysis = analyze_description(query)

        print(f"\nQuery: {query}")
        print(f"Route: {analysis.route}")
        print(f"Focus: {analysis.focus}")

        if analysis.route == "unspecified":
            print("Description không phù hợp để tìm species.")
            return

        image_results = search_by_description(db, query, top_k=5)
        text_results = search_species_knowledge(db, query, top_k=5)

        fast_results = hybrid_search(
            db, query, analysis.focus, search_mode="fast", top_k=5
        )

        deep_results = hybrid_search(
            db, query, analysis.focus, search_mode="deep", top_k=5
        )

        print_results("IMAGE RETRIEVAL", image_results, "similarity")
        print_results("TEXT RETRIEVAL", text_results, "knowledge_score")
        print_results("HYBRID FAST", fast_results, "score")
        print_results("HYBRID DEEP", deep_results, "score")

    finally:
        db.close()


if __name__ == "__main__":
    main()