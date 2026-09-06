import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.message_router_service import classify_message


QUESTIONS = [
    "Rắn cạp nong có độc không?",
    "Bị rắn cắn nên làm gì?",
    "Rắn lục đầu bạc sống ở đâu?",
    "Chào bạn",
    "Trời hôm nay đẹp nhỉ?",
    "Bạn khỏe không?",
]


def main():
    for index, question in enumerate(QUESTIONS, 1):
        route = classify_message(question)

        print(f"{index}. {question}")
        print(f"   ROUTE: {route}\n")


if __name__ == "__main__":
    main()