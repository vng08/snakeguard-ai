import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from backend.app.db.session import SessionLocal
from backend.app.services.llm_service import generate_answer
from backend.app.services.retrieval_service import search_knowledge


QUESTIONS = [
    "Rắn cạp nong có đặc điểm nhận dạng như thế nào?",
    "Bị rắn cắn nên sơ cứu như thế nào?",
    "Rắn cạp nong cắn có triệu chứng gì?",
    "Trời hôm nay đẹp nhỉ?",
]


def main():
    db = SessionLocal()

    try:
        for index, question in enumerate(QUESTIONS, 1):
            retrieval = search_knowledge(db, question, top_k=5, use_reranker=True)
            result = generate_answer(db, question, retrieval["results"])

            print(f"\n{'=' * 80}")
            print(f"QUESTION {index}: {question}")
            print(f"\nCAN ANSWER: {result['can_answer']}")
            print(f"ANSWER:\n{result['answer']}")
    finally:
        db.close()


if __name__ == "__main__":
    main()