from pydantic import BaseModel, ConfigDict


class SnakeSpeciesResponse(BaseModel):
    """Schema trả thông tin loài rắn từ database."""

    id: int
    binomial_name: str
    vietnamese_name: str | None = None
    family: str | None = None
    genus: str | None = None
    is_mivs: bool

    model_config = ConfigDict(from_attributes=True)