from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.db.models import ChatMessage


def save_chat_message(db: Session, session_id: str, role: str, content: str) -> ChatMessage:
    """Lưu một message vào lịch sử hội thoại."""
    message = ChatMessage(session_id=session_id, role=role, content=content)
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_chat_history(db: Session, session_id: str) -> list[ChatMessage]:
    """Lấy lịch sử conversation hiện tại theo thứ tự thời gian."""
    return db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc()).all()


def delete_chat_history(db: Session) -> int:
    """Xoá toàn bộ lịch sử chat và reset message ID."""
    deleted_count = db.query(ChatMessage).count()
    db.execute(text("TRUNCATE TABLE chat_messages RESTART IDENTITY"))
    db.commit()
    return deleted_count