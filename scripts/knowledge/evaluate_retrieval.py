import sys
from pathlib import Path

# Cho phép import backend khi chạy trực tiếp script
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from backend.app.db.session import SessionLocal
from backend.app.services.retrieval_service import search_knowledge


TOP_K = 5

# Gold set ban đầu, có thể bổ sung dần lên 30–50 câu
EVAL_QUERIES = [
    # Distribution
    {
        "query": "Trimeresurus albolabris phân bố ở đâu?",
        "scope": "Trimeresurus albolabris",
        "section": "distribution",
    },
    {
        "query": "Rắn cạp nong phân bố ở những khu vực nào?",
        "scope": "Bungarus fasciatus",
        "section": "distribution",
    },
    {
        "query": "Trimeresurus vogeli được tìm thấy ở đâu?",
        "scope": "Trimeresurus vogeli",
        "section": "distribution",
    },

    # Identification
    {
        "query": "Đặc điểm nhận dạng của Bungarus fasciatus?",
        "scope": "Bungarus fasciatus",
        "section": "identification",
    },
    {
        "query": "Rắn cạp nong có đặc điểm nhận dạng như thế nào?",
        "scope": "Bungarus fasciatus",
        "section": "identification",
    },
    {
        "query": "Rắn hổ chúa có đặc điểm hình dạng gì?",
        "scope": "Ophiophagus hannah",
        "section": "identification",
    },

    # Habitat / behavior
    {
        "query": "Bungarus fasciatus có tập tính sinh sống như thế nào?",
        "scope": "Bungarus fasciatus",
        "section": "habitat_behavior",
    },
    {
        "query": "Trimeresurus albolabris thường sống và hoạt động ở đâu?",
        "scope": "Trimeresurus albolabris",
        "section": "habitat_behavior",
    },
    {
        "query": "Rắn hổ hành có tập tính như thế nào?",
        "scope": "Xenopeltis unicolor",
        "section": "habitat_behavior",
    },

    # Venom / danger
    {
        "query": "Rắn hổ chúa có nguy hiểm không?",
        "scope": "Ophiophagus hannah",
        "section": "venom_danger",
    },
    {
        "query": "Rắn bông súng có độc không?",
        "scope": "Enhydris enhydris",
        "section": "venom_danger",
    },

    # Diagnosis / symptoms
    {
        "query": "Rắn cạp nia cắn có triệu chứng gì?",
        "scope": "Bungarus",
        "section": "diagnosis",
    },
    {
        "query": "Bị rắn hổ mang cắn có những dấu hiệu gì?",
        "scope": "Naja atra; Naja kaouthia",
        "section": "diagnosis",
    },
    {
        "query": "Rắn hổ mèo cắn có triệu chứng gì?",
        "scope": "Naja siamensis",
        "section": "diagnosis",
    },
    {
        "query": "Rắn chàm quạp cắn có biểu hiện gì?",
        "scope": "Calloselasma rhodostoma",
        "section": "diagnosis",
    },
    {
        "query": "Bị rắn lục cắn thường có những triệu chứng nào?",
        "scope": "Viperidae",
        "section": "diagnosis",
    },

    # Treatment
    {
        "query": "Rắn hổ mang cắn được điều trị như thế nào?",
        "scope": "Naja atra; Naja kaouthia",
        "section": "treatment",
    },
    {
        "query": "Điều trị khi bị rắn lục cắn như thế nào?",
        "scope": "Viperidae",
        "section": "treatment",
    },

    # General first aid
    {
        "query": "Bị rắn cắn nên sơ cứu như thế nào?",
        "scope": None,
        "section": "first_aid",
    },
    {
        "query": "Nếu không biết loại rắn vừa cắn thì nên sơ cứu ra sao?",
        "scope": None,
        "section": "first_aid",
    },
]


def is_relevant(result, gold):
    """Chunk đúng khi đúng section và đúng scope nếu gold có scope."""
    if result["section"] != gold["section"]:
        return False

    if gold["scope"] is None:
        return True

    return result["scope_value"] == gold["scope"]


def reciprocal_rank(results, gold):
    """RR = 1 / vị trí chunk đúng đầu tiên."""
    for rank, result in enumerate(results, 1):
        if is_relevant(result, gold):
            return 1 / rank
    return 0.0


def hit_at_k(results, gold):
    """Có ít nhất một chunk đúng trong Top-K hay không."""
    return int(any(is_relevant(result, gold) for result in results))


def precision_at_k(results, gold):
    """Tỷ lệ chunk đúng trong Top-K."""
    if not results:
        return 0.0

    relevant = sum(is_relevant(result, gold) for result in results)
    return relevant / len(results)


def evaluate(db, use_reranker):
    """Evaluate toàn bộ gold queries."""
    hits, reciprocal_ranks, precisions = [], [], []

    print(f"\n{'WITH RERANKER' if use_reranker else 'WITHOUT RERANKER'}")
    print("=" * 70)

    for item in EVAL_QUERIES:
        results = search_knowledge(
            db,
            item["query"],
            top_k=TOP_K,
            use_reranker=use_reranker,
        )

        hit = hit_at_k(results, item)
        rr = reciprocal_rank(results, item)
        precision = precision_at_k(results, item)

        hits.append(hit)
        reciprocal_ranks.append(rr)
        precisions.append(precision)

        first = results[0] if results else None
        top1 = (
            f"{first['scope_value']} | {first['section']}"
            if first else "None"
        )

        print(
            f"\nQuery: {item['query']}\n"
            f"Hit@{TOP_K}: {hit} | RR: {rr:.3f} | "
            f"P@{TOP_K}: {precision:.3f}\n"
            f"Top1: {top1}"
        )

    return {
        f"Hit@{TOP_K}": sum(hits) / len(hits),
        f"MRR@{TOP_K}": sum(reciprocal_ranks) / len(reciprocal_ranks),
        f"Precision@{TOP_K}": sum(precisions) / len(precisions),
    }


def main():
    db = SessionLocal()

    try:
        before = evaluate(db, use_reranker=False)
        after = evaluate(db, use_reranker=True)
    finally:
        db.close()

    print("\n\nFINAL RESULTS")
    print("=" * 50)
    print(f"{'Metric':<15} {'Before':>10} {'After':>10}")

    for metric in before:
        print(f"{metric:<15} {before[metric]:>10.3f} {after[metric]:>10.3f}")


if __name__ == "__main__":
    main()