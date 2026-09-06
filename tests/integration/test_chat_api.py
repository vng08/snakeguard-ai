from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.app.api.chat as chat_api
from backend.app.api.dependencies import get_db


app = FastAPI()
app.include_router(chat_api.router)
app.dependency_overrides[get_db] = lambda: object()
client = TestClient(app)


def test_chat_knowledge_base_response(monkeypatch):
    """KB response -> HTTP 200 và schema đúng."""
    monkeypatch.setattr(chat_api, "chat", lambda db, message: {
        "answer": "Rắn cạp nong có các khoang vàng đen rõ rệt.",
        "source_type": "knowledge_base",
        "sources": [{"source": "VietnamSnakes", "source_url": "https://example.com", "section": "identification", "scope_value": "Bungarus fasciatus"}],
        "is_ambiguous": False,
        "needs_clarification": False,
        "candidate_species": [],
        "notice": None,
    })

    response = client.post("/api/chat", json={"message": "Rắn cạp nong có đặc điểm gì?"})
    data = response.json()

    assert response.status_code == 200
    assert data["source_type"] == "knowledge_base"
    assert data["answer"] == "Rắn cạp nong có các khoang vàng đen rõ rệt."
    assert data["needs_clarification"] is False
    assert len(data["sources"]) == 1


def test_chat_casual_response(monkeypatch):
    """Casual response -> HTTP 200."""
    monkeypatch.setattr(chat_api, "chat", lambda db, message: {
        "answer": "Ừ, nghe cũng dễ chịu đấy 😄",
        "source_type": "casual",
        "sources": [],
        "is_ambiguous": False,
        "needs_clarification": False,
        "candidate_species": [],
        "notice": None,
    })

    response = client.post("/api/chat", json={"message": "Trời hôm nay đẹp nhỉ?"})
    data = response.json()

    assert response.status_code == 200
    assert data["source_type"] == "casual"
    assert data["sources"] == []
    assert data["notice"] is None


def test_chat_unavailable_response(monkeypatch):
    """Web unavailable -> API vẫn HTTP 200, không 500."""
    monkeypatch.setattr(chat_api, "chat", lambda db, message: {
        "answer": "Cơ sở kiến thức hiện tại chưa đủ để trả lời câu hỏi này và tính năng tìm kiếm Internet đang tạm thời không khả dụng. Bạn có thể thử lại sau.",
        "source_type": "unavailable",
        "sources": [],
        "is_ambiguous": False,
        "needs_clarification": False,
        "candidate_species": [],
        "notice": "Tìm kiếm Internet hiện không khả dụng.",
    })

    response = client.post("/api/chat", json={"message": "Rắn lạ này sống ở đâu?"})
    data = response.json()

    assert response.status_code == 200
    assert data["source_type"] == "unavailable"
    assert data["sources"] == []
    assert data["notice"] == "Tìm kiếm Internet hiện không khả dụng."


def test_chat_missing_message_returns_422():
    """Request thiếu message -> FastAPI validation trả 422."""
    response = client.post("/api/chat", json={})

    assert response.status_code == 422