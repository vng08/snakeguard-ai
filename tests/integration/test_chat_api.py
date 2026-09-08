from datetime import datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.app.api import chat as chat_api
from backend.app.api.dependencies import get_db
from backend.app.main import app


client = TestClient(app)


class FakeDB:
    """Giả lập database session cho Chat API."""
    pass


def override_get_db():
    """Trả FakeDB để test không kết nối database thật."""
    yield FakeDB()


app.dependency_overrides[get_db] = override_get_db


def test_send_message(monkeypatch):
    """Test POST /chat trả answer và sources."""
    def mock_chat(db, session_id, message, search_mode):
        return {
            "session_id": session_id,
            "answer": "Rắn hổ đất là loài rắn độc.",
            "sources": [{"source": "VietnamSnakes", "source_url": "https://example.com"}],
        }

    monkeypatch.setattr(chat_api, "chat", mock_chat)

    response = client.post("/chat", json={
        "session_id": "test-session",
        "message": "Rắn hổ đất có độc không?",
        "search_mode": "fast",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test-session"
    assert data["answer"] == "Rắn hổ đất là loài rắn độc."
    assert len(data["sources"]) == 1


def test_get_chat_history(monkeypatch):
    """Test GET /chat/{session_id} trả lịch sử conversation."""
    messages = [
        SimpleNamespace(id=1, session_id="test-session", role="user", content="Xin chào", created_at=datetime(2026, 9, 8, 10, 0)),
        SimpleNamespace(id=2, session_id="test-session", role="assistant", content="Chào bạn!", created_at=datetime(2026, 9, 8, 10, 1)),
    ]

    monkeypatch.setattr(chat_api, "get_chat_history", lambda db, session_id: messages)

    response = client.get("/chat/test-session")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test-session"
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][1]["role"] == "assistant"


def test_delete_chat_history(monkeypatch):
    """Test DELETE /chat xoá conversation hiện tại."""
    monkeypatch.setattr(chat_api, "delete_chat_history", lambda db: 4)

    response = client.delete("/chat")

    assert response.status_code == 200
    assert response.json() == {"deleted_count": 4}


def test_send_message_runtime_error(monkeypatch):
    """Test POST /chat trả 500 khi Agentic RAG không tạo được answer."""
    def mock_chat(*args, **kwargs):
        raise RuntimeError("Agentic RAG không tạo được câu trả lời.")

    monkeypatch.setattr(chat_api, "chat", mock_chat)

    response = client.post("/chat", json={
        "session_id": "test-session",
        "message": "Hello",
        "search_mode": "fast",
    })

    assert response.status_code == 500
    assert response.json()["detail"] == "Agentic RAG không tạo được câu trả lời."