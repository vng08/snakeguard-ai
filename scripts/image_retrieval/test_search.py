from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from backend.app.db.session import SessionLocal
from backend.app.services.image_retrieval_service import search_by_description
from backend.app.services.hybrid_retrieval_service import hybrid_search, search_species_knowledge


def print_results(title, results, score_key):
    print(f"\n=== {title} ===")

    for i, result in enumerate(results, 1):
        print(
            f"{i}. {result['binomial_name']} | "
            f"{result['vietnamese_name']} | "
            f"score={result[score_key]:.4f}"
        )
        print(f"   image: {result.get('image_url')}")


def main():
    db = SessionLocal()
    query = "rắn có lưng có các đốm sậm màu nối liền với đường sống lưng so le nhau"

    try:
        image_results = search_by_description(db, query, top_k=5)
        text_results = search_species_knowledge(db, query, top_k=5)
        hybrid_results = hybrid_search(db, query, top_k=5)

        print(f"\nQuery: {query}")

        print_results("IMAGE RETRIEVAL", image_results, "similarity")
        print_results("TEXT RETRIEVAL", text_results, "knowledge_score")
        print_results("HYBRID RETRIEVAL", hybrid_results, "hybrid_score")

    finally:
        db.close()


if __name__ == "__main__":
    main()