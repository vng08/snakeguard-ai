from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from backend.app.services.rag.nodes.answer_evaluator import evaluate_answer
from backend.app.services.rag.nodes.answer_generator import generate_answer
from backend.app.services.rag.nodes.query_analyzer import analyze_query
from backend.app.services.rag.orchestration.state import ChatState
from backend.app.services.rag.tools.reranker import rerank_documents
from backend.app.services.rag.tools.retrieval import retrieve_knowledge
from backend.app.services.rag.tools.species_resolver import resolve_species
from backend.app.services.rag.tools.web_search import search_web


def build_chat_graph(db: Session):
    """Khởi tạo LangGraph workflow cho Agentic RAG."""
    graph = StateGraph(ChatState)

    graph.add_node("query_analyzer", analyze_query)
    graph.add_node("species_resolver", lambda state: _resolve_species_node(db, state))
    graph.add_node("retrieval", lambda state: _retrieval_node(db, state))
    graph.add_node("reranker", _reranker_node)
    graph.add_node("evaluator", evaluate_answer)
    graph.add_node("web_search", search_web)
    graph.add_node("answer_generator", generate_answer)
    graph.add_node("clarify_species", _clarify_species_node)

    graph.add_edge(START, "query_analyzer")
    graph.add_conditional_edges("query_analyzer", _route_after_analysis, {"casual": "answer_generator", "global": "retrieval", "species": "species_resolver", "identify": "answer_generator"})
    graph.add_conditional_edges("species_resolver", _route_after_species_resolution, {"resolved": "retrieval", "ambiguous": "clarify_species", "not_found": "web_search"})
    graph.add_conditional_edges("retrieval", _route_after_retrieval, {"fast": "evaluator", "deep": "reranker"})
    graph.add_edge("reranker", "evaluator")
    graph.add_conditional_edges("evaluator", _route_after_evaluation, {"answer": "answer_generator", "web": "web_search"})
    graph.add_edge("web_search", "evaluator")
    graph.add_edge("answer_generator", END)
    graph.add_edge("clarify_species", END)

    return graph.compile()


def _resolve_species_node(db: Session, state: ChatState) -> dict:
    """Resolve species được Query Analyzer phát hiện."""
    species_text = state["analysis"].get("species_text")

    if not species_text:
        return {"species_status": "not_found", "species_candidates": [], "resolved_species_ids": []}

    result = resolve_species(db, species_text, state["search_mode"])
    return {"species_status": result["status"], "species_candidates": result["candidates"], "resolved_species_ids": result["resolved_species_ids"]}


def _retrieval_node(db: Session, state: ChatState) -> dict:
    """Retrieve knowledge theo route và species đã resolve."""
    analysis = state["analysis"]
    species = state["species_candidates"][0] if analysis["route"] == "species" else None

    documents = retrieve_knowledge(
        db=db,
        query=analysis["standalone_query"],
        route=analysis["route"],
        species=species,
        search_mode=state["search_mode"],
    )

    return {"kb_documents": documents}


def _reranker_node(state: ChatState) -> dict:
    """Rerank knowledge candidates trong Deep mode."""
    query = state["analysis"]["standalone_query"]
    documents = rerank_documents(query, state.get("kb_documents", []))
    return {"kb_documents": documents}


def _clarify_species_node(state: ChatState) -> dict:
    """Yêu cầu người dùng làm rõ khi tên rắn khớp nhiều species."""
    candidates = state.get("species_candidates", [])[:5]
    names = [f"- {candidate['matched_name']} ({candidate['binomial_name']})" for candidate in candidates]
    candidate_text = "\n".join(names)

    answer = f"Tên rắn bạn cung cấp có thể khớp với nhiều loài khác nhau:\n\n{candidate_text}\n\nBạn có thể cho mình biết chính xác loài bạn đang muốn hỏi không?"
    return {"answer": answer, "sources": []}


def _route_after_analysis(state: ChatState) -> str:
    """Chọn workflow dựa trên route từ Query Analyzer."""
    return state["analysis"]["route"]


def _route_after_species_resolution(state: ChatState) -> str:
    """Chọn workflow dựa trên kết quả Species Resolver."""
    return state["species_status"]


def _route_after_retrieval(state: ChatState) -> str:
    """Deep mode rerank documents, Fast mode đánh giá trực tiếp."""
    return state["search_mode"]


def _route_after_evaluation(state: ChatState) -> str:
    """Quyết định trả lời hoặc fallback sang web search."""
    if state["evaluation"]["sufficient"]:
        return "answer"

    if "web_documents" in state:
        return "answer"

    return "web"