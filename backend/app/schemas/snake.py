from pydantic import BaseModel


class SnakeSpeciesCreate(BaseModel):
    binomial_name: str
    vietnamese_name: str | None = None
    family: str | None = None
    genus: str | None = None
    is_mivs: bool


class SnakeSpeciesResponse(SnakeSpeciesCreate):
    id: int

    class Config:
        from_attributes = True