import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from backend.app.db.session import SessionLocal
from backend.app.services.rag.tools.retrieval import retrieve_knowledge
from backend.app.services.rag.tools.reranker import rerank_documents
from backend.app.services.rag.tools.species_resolver import resolve_species


def print_resolution(title: str, resolution: dict):
    """In kết quả Species Resolver và các score dùng để quyết định."""
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)
    print("Status:", resolution["status"])
    print("Resolved IDs:", resolution["resolved_species_ids"])

    for index, candidate in enumerate(resolution["candidates"], start=1):
        print(
            f"\n[{index}] {candidate['binomial_name']} | matched={candidate['matched_name']} "
            f"| type={candidate['match_type']} | score={candidate['score']}"
        )

        if candidate.get("lexical_score") is not None:
            print(f"    lexical_score={candidate['lexical_score']}")

        if candidate.get("semantic_score") is not None:
            print(f"    semantic_score={candidate['semantic_score']}")

        if candidate.get("match_score") is not None:
            print(f"    match_score={candidate['match_score']}")

        if candidate.get("rerank_score") is not None:
            print(f"    rerank_score={candidate['rerank_score']}")

        if candidate.get("final_score") is not None:
            print(f"    final_score={candidate['final_score']}")


def print_results(title: str, results: list[dict]):
    """In kết quả retrieval để kiểm tra score và taxonomy scope."""
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)

    for index, item in enumerate(results, start=1):
        rerank_score = f" | rerank={item['rerank_score']:.4f}" if "rerank_score" in item else ""

        print(
            f"\n[{index}] score={item['vector_score']:.4f}{rerank_score} | level={item['match_level']} "
            f"| section={item['section']} | scope={item['scope_type']}:{item['scope_value']}"
        )
        print(f"Source: {item['source']}")
        print(item["content"][:500])


def test_species_medical(db):
    """Test Fast Search với câu hỏi y khoa của một loài cụ thể."""
    query = "Rắn hổ đất cắn thì nên xử lý như thế nào?"

    resolution = resolve_species(db, "Rắn hổ đất", search_mode="fast")
    print_resolution("FAST SPECIES RESOLUTION", resolution)

    if resolution["status"] != "resolved":
        print("Không resolve được species.")
        return

    species = resolution["candidates"][0]
    results = retrieve_knowledge(db=db, query=query, route="species", species=species, search_mode="fast")

    print_results(f"FAST - {query}", results)


def test_species_identification(db):
    """Test Fast Search với câu hỏi đặc điểm của một loài cụ thể."""
    query = "Rắn hổ đất có đặc điểm nhận dạng gì?"

    resolution = resolve_species(db, "Rắn hổ đất", search_mode="fast")

    if resolution["status"] != "resolved":
        print("Không resolve được species.")
        return

    species = resolution["candidates"][0]
    results = retrieve_knowledge(db=db, query=query, route="species", species=species, search_mode="fast")

    print_results(f"FAST - {query}", results)


def test_global_first_aid(db):
    """Test Fast Search với câu hỏi chung không yêu cầu resolve species."""
    query = "Bị rắn cắn thì nên sơ cứu như thế nào?"

    results = retrieve_knowledge(db=db, query=query, route="global", species=None, search_mode="fast")
    print_results(f"FAST - {query}", results)


def test_species_resolution_deep(db):
    """Test Deep Species Resolver khi lexical matching không đủ."""
    species_text = "monocled cobra"

    resolution = resolve_species(db, species_text, search_mode="deep")
    print_resolution(f"DEEP SPECIES RESOLVER - {species_text}", resolution)


def test_species_medical_deep(db):
    """Test Deep retrieval sau khi Ratio resolve species."""
    query = "Rắn hổ đất cắn thì nên xử lý như thế nào?"

    resolution = resolve_species(db, "Rắn hổ dap", search_mode="deep")
    print_resolution("DEEP RETRIEVAL SPECIES RESOLUTION", resolution)

    if resolution["status"] != "resolved":
        print("Không resolve được species.")
        return

    species = resolution["candidates"][0]

    # Deep retrieval lấy candidate pool lớn trước khi rerank
    candidates = retrieve_knowledge(db=db, query=query, route="species", species=species, search_mode="deep")
    print_results(f"DEEP CANDIDATES - {query}", candidates)

    # Reranker chọn lại các knowledge chunks phù hợp nhất
    results = rerank_documents(query, candidates)
    print_results(f"DEEP RERANKED - {query}", results)


def main():
    db = SessionLocal()

    try:
        test_species_medical(db)
        test_species_identification(db)
        test_global_first_aid(db)
        test_species_resolution_deep(db)
        test_species_medical_deep(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()