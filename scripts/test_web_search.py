import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.web_search_service import search_web


def main():
    question = "Rắn lục đầu bạc Azemiops feae sống ở đâu và có độc không?"
    result = search_web(question)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print(f"- {source['source']}")
        print(f"  {source['source_url']}")


if __name__ == "__main__":
    main()