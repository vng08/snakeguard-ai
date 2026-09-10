import json
from functools import lru_cache

from openai import OpenAI

from backend.app.core.config import settings

PROVIDER_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}


@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI:
    """Khởi tạo và cache LLM client theo provider trong cấu hình."""
    provider = settings.LLM_PROVIDER.lower().strip()

    if provider == "openai":
        return OpenAI(api_key=settings.LLM_API_KEY)

    if provider in PROVIDER_BASE_URLS:
        return OpenAI(api_key=settings.LLM_API_KEY, base_url=PROVIDER_BASE_URLS[provider])

    raise ValueError(
        f"LLM provider không được hỗ trợ: {settings.LLM_PROVIDER}. Hỗ trợ: groq, openai, deepseek, gemini."
    )


def generate_text(messages: list[dict], temperature: float = 0) -> str:
    """Gọi LLM và trả về nội dung dạng text."""
    response = get_llm_client().chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def generate_json(messages: list[dict], temperature: float = 0) -> dict:
    """Gọi LLM ở JSON mode và trả về dictionary."""
    response = get_llm_client().chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=temperature,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM trả về nội dung JSON rỗng.")

    return json.loads(content)