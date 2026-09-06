from pydantic import BaseModel, Field


class SearchByDescriptionRequest(BaseModel):
    description: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=10)


class SearchResult(BaseModel):
    species_id: int
    binomial_name: str
    vietnamese_name: str | None = None
    image_url: str | None = None

    image_score_normalized: float
    knowledge_score_normalized: float
    hybrid_score: float
    is_mivs: bool


class SearchByDescriptionResponse(BaseModel):
    description: str
    results: list[SearchResult]