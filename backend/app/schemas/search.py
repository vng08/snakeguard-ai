from typing import Literal

from pydantic import BaseModel, Field


class DescriptionAnalysis(BaseModel):
    """Kết quả phân tích description phục vụ hybrid search."""
    route: Literal["snake", "unspecified"]
    focus: Literal["visual", "mixed", "contextual"]


class SearchByDescriptionRequest(BaseModel):
    """Request tìm species từ mô tả."""
    description: str = Field(..., min_length=3)
    search_mode: Literal["fast", "deep"] = "fast"
    top_k: int = Field(default=5, ge=1, le=10)


class SearchResult(BaseModel):
    """Một species được tìm thấy từ description."""
    species_id: int
    binomial_name: str
    vietnamese_name: str | None = None
    image_url: str | None = None
    is_mivs: bool
    score: float


class SearchByDescriptionResponse(BaseModel):
    """Response của Search by Description."""
    description: str
    results: list[SearchResult]