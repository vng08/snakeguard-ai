from pydantic import BaseModel
from backend.app.schemas.snake import SnakeSpeciesResponse

class SpeciesPrediction(BaseModel):
    label_idx: int
    class_id: int
    binomial_name: str
    confidence: float
    MIVS: int

class DetectionInfo(BaseModel):
    bbox: list[int]
    confidence: float

class PredictionResponse(BaseModel):
    detected: bool
    detection: DetectionInfo | None = None
    predictions: list[SpeciesPrediction] = []
    species: SnakeSpeciesResponse | None = None