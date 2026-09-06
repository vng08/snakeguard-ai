from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class ChatSource(BaseModel):
    source: str
    source_url: str | None = None
    section: str | None = None
    scope_value: str | None = None


class ChatResponse(BaseModel):
    answer: str
    source_type: str
    sources: list[ChatSource]
    is_ambiguous: bool
    needs_clarification: bool
    candidate_species: list[str]
    notice: str | None = None