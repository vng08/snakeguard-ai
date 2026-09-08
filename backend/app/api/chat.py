from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db
from backend.app.schemas.chat import ChatHistoryResponse, ChatRequest, ChatResponse
from backend.app.services.rag.memory.chat_history_service import delete_chat_history, get_chat_history
from backend.app.services.rag.orchestration.chat_service import chat


router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
def send_message(request: ChatRequest, db: Session = Depends(get_db)):
    """Gửi một message vào Agentic RAG chatbot."""
    try:
        return chat(db, request.session_id, request.message, request.search_mode)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/chat/{session_id}", response_model=ChatHistoryResponse)
def get_history(session_id: str, db: Session = Depends(get_db)):
    """Lấy lịch sử conversation hiện tại."""
    messages = get_chat_history(db, session_id)
    return {"session_id": session_id, "messages": messages}


@router.delete("/chat")
def delete_history(db: Session = Depends(get_db)):
    """Xoá conversation hiện tại và reset message ID."""
    deleted_count = delete_chat_history(db)
    return {"deleted_count": deleted_count}