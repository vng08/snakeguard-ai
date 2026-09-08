from groq import Groq

from backend.app.core.config import settings


def get_llm_client() -> Groq:
    """Khởi tạo Groq client dùng chung cho các node RAG."""
    return Groq(api_key=settings.GROQ_API_KEY)