from pydantic import BaseModel


class SnakeSpeciesCreate(BaseModel):
    scientific_name: str
    vietnamese_name: str | None = None
    common_name: str | None = None
    family: str
    genus: str
    venomous: bool


class SnakeSpeciesResponse(SnakeSpeciesCreate):
    id: int

    class Config:
        from_attributes = True