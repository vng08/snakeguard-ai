from datetime import datetime
from pydantic import BaseModel, ConfigDict
from backend.app.schemas.snake import SnakeSpeciesResponse


class SpeciesPrediction(BaseModel):
    """Kết quả classification của một loài rắn."""
    label_idx: int
    class_id: int
    binomial_name: str
    confidence: float
    MIVS: int


class DetectionInfo(BaseModel):
    """Thông tin bounding box và độ tin cậy detection."""
    bbox: list[int]
    confidence: float


class PredictionResponse(BaseModel):
    """Response trả về từ API nhận diện rắn."""
    detected: bool
    detection: DetectionInfo | None = None
    predictions: list[SpeciesPrediction] = []
    species: SnakeSpeciesResponse | None = None


class PredictionLogResponse(BaseModel):
    """Response trả thông tin một prediction log."""
    id: int
    image_url: str | None = None
    bbox: list[int] | None = None
    detection_confidence: float | None = None
    predicted_species_id: int | None = None
    confidence: float | None = None
    top_k_predictions: list[SpeciesPrediction] | None = None
    model_version: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)