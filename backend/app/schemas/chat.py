from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class QueryAnalysis(BaseModel):
    """Kết quả phân tích câu hỏi dựa trên context hội thoại."""

    route: Literal["casual", "global", "species"]
    species_text: str | None = None
    standalone_query: str


class AnswerEvaluation(BaseModel):
    """Kết quả đánh giá mức độ đầy đủ của context để trả lời câu hỏi."""

    sufficient: bool
    confidence: float
    missing_information: str | None = None


class ChatRequest(BaseModel):
    """Request gửi một message vào chatbot."""

    session_id: str
    message: str
    search_mode: Literal["fast", "deep"] = "fast"


class ChatSource(BaseModel):
    """Nguồn được sử dụng để tạo câu trả lời."""

    source: str
    source_url: str | None = None


class ChatResponse(BaseModel):
    """Response trả về sau một lượt hội thoại."""

    session_id: str
    answer: str
    sources: list[ChatSource] = []


class ChatMessageResponse(BaseModel):
    """Response của một message trong lịch sử chat."""

    id: int
    session_id: str
    role: str
    content: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class ChatHistoryResponse(BaseModel):
    """Response trả toàn bộ conversation hiện tại."""

    session_id: str
    messages: list[ChatMessageResponse] = []