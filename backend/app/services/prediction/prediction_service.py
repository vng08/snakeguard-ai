from PIL import Image
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.models import SnakeSpecies
from backend.app.schemas.prediction import PredictionResponse
from backend.app.schemas.snake import SnakeSpeciesResponse
from backend.app.services.media.storage_service import save_prediction_image
from backend.app.services.prediction.prediction_log_service import save_prediction_log
from ml.pipeline import SnakeGuardPipeline


class PredictionService:
    """Điều phối toàn bộ logic của một lần nhận diện rắn."""

    def __init__(self, pipeline: SnakeGuardPipeline):
        self.pipeline = pipeline

    def predict(self, db: Session, image: Image.Image, top_k: int = 3) -> PredictionResponse:
        """Chạy pipeline, lưu kết quả và trả về prediction response."""
        results = self.pipeline.predict(image, top_k=top_k)
        primary = max(results, key=lambda item: item["detection_confidence"]) if results else None

        # Lưu ảnh và lịch sử prediction
        image_url = save_prediction_image(image)
        save_prediction_log(db, primary, image_url, settings.MODEL_VERSION)

        if not primary:
            return PredictionResponse(detected=False)

        predictions = primary.get("predictions", [])
        species = self._get_predicted_species(db, predictions)

        return PredictionResponse(
            detected=True,
            detection={
                "bbox": primary["bbox"],
                "confidence": primary["detection_confidence"],
            },
            predictions=predictions,
            species=species,
        )

    def _get_predicted_species(self, db: Session, predictions: list[dict]) -> SnakeSpeciesResponse | None:
        """Lấy thông tin loài tương ứng với kết quả classification Top-1."""
        if not predictions:
            return None

        binomial_name = predictions[0]["binomial_name"]
        species = db.query(SnakeSpecies).filter(SnakeSpecies.binomial_name == binomial_name).first()

        return SnakeSpeciesResponse.model_validate(species) if species else None