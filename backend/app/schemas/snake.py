from pydantic import BaseModel, ConfigDict

class SnakeSpeciesCreate(BaseModel):
    binomial_name: str
    vietnamese_name: str | None = None
    family: str | None = None
    genus: str | None = None
    is_mivs: bool

class SnakeSpeciesResponse(SnakeSpeciesCreate):
    id: int
    model_config = ConfigDict(from_attributes=True)