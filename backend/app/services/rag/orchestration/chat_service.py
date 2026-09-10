from typing import Literal

from sqlalchemy.orm import Session

from backend.app.services.rag.memory.chat_history_service import (
    get_chat_history,
    save_chat_message,
)
from backend.app.services.rag.orchestration.graph import build_chat_graph
from backend.app.services.rag.orchestration.state import ChatState


def chat(db: Session, session_id: str, message: str, search_mode: Literal["fast", "deep"] = "fast") -> dict:
    """Xử lý một lượt hội thoại qua Agentic RAG."""
    history = get_chat_history(db, session_id)

    state: ChatState = {
        "session_id": session_id,
        "message": message,
        "history": history,
        "search_mode": search_mode,
    }

    graph = build_chat_graph(db)
    result = graph.invoke(state)
    answer = result.get("answer")

    if not answer:
        raise RuntimeError("Agentic RAG không tạo được câu trả lời.")

    # Chỉ lưu message sau khi workflow xử lý thành công
    save_chat_message(db, session_id, "user", message)
    save_chat_message(db, session_id, "assistant", answer)

    return {"session_id": session_id, "answer": answer, "sources": result.get("sources", [])}