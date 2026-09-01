from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session
from backend.app.api.dependencies import get_db
from backend.app.core.config import settings
from backend.app.db.models import SnakeSpecies
from backend.app.schemas.prediction import PredictionResponse
from backend.app.schemas.snake import SnakeSpeciesResponse
from backend.app.services.image_service import decode_image
from backend.app.services.prediction_log_service import save_prediction_log
from backend.app.services.prediction_service import PredictionService
from backend.app.services.storage_service import save_prediction_image

router = APIRouter(tags=["Prediction"])
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}

@router.post("/predict", response_model=PredictionResponse)
async def predict_snake(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported image type. Allowed: JPG, JPEG, PNG, WEBP, BMP, TIFF.")

        image = decode_image(await file.read())
        results = PredictionService(request.app.state.pipeline).predict(image, top_k=3)
        primary = max(results, key=lambda x: x["detection_confidence"]) if results else None
        image_url = save_prediction_image(image)
        save_prediction_log(db, primary, image_url, settings.MODEL_VERSION)

        if not primary:
            return {"detected": False, "detection": None, "predictions": [], "species": None}

        top1 = primary["predictions"][0]
        species = db.query(SnakeSpecies).filter(SnakeSpecies.binomial_name == top1["binomial_name"]).first()

        return {
            "detected": True,
            "detection": {"bbox": primary["bbox"], "confidence": primary["detection_confidence"]},
            "predictions": primary["predictions"],
            "species": SnakeSpeciesResponse.model_validate(species) if species else None,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc