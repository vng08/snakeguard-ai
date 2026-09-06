from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.chat_service import chat


router = APIRouter(prefix="/api/chat", tags=["Chatbot"])


@router.post("", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    return chat(db, request.message)