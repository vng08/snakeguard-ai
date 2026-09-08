from typing import Literal, NotRequired, TypedDict


class ChatState(TypedDict):
    """State được truyền xuyên suốt workflow Agentic RAG."""

    # Dữ liệu đầu vào
    session_id: str
    message: str
    history: list
    search_mode: Literal["fast", "deep"]

    # Kết quả từ Query Analyzer
    analysis: NotRequired[dict]

    # Kết quả từ Species Resolver
    species_status: NotRequired[Literal["resolved", "ambiguous", "not_found"]]
    species_candidates: NotRequired[list[dict]]
    resolved_species_ids: NotRequired[list[int]]

    # Context retrieval
    kb_documents: NotRequired[list[dict]]
    web_documents: NotRequired[list[dict]]

    # Kết quả đánh giá context
    evaluation: NotRequired[dict]

    # Kết quả cuối cùng
    answer: NotRequired[str]
    sources: NotRequired[list[dict]]